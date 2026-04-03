# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.lerobot_import_utils import import_lerobot_policy
from rlinf.models.embodiment.modules.compact_encoders import (
    CompactMultiQHead,
    CompactStateEncoder,
    LightweightImageEncoder64,
)
from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy
from rlinf.models.embodiment.modules.value_head import ValueHead
import logging

logger = logging.getLogger(__name__)


@dataclass
class SmolVLAConfig:
    state_dim: int
    action_dim: int
    num_action_chunks: int
    hidden_dim: int = 512
    image_size: list[int] = field(default_factory=lambda: [3, 224, 224])
    add_value_head: bool = False
    min_std: float = 1e-4
    max_std: float = 10.0
    lerobot_src_path: str | None = None
    use_dsrl: bool = False
    dsrl_state_dim: int = 14
    dsrl_action_noise_dim: int = 14
    dsrl_num_q_heads: int = 2
    dsrl_image_latent_dim: int = 64
    dsrl_state_latent_dim: int = 64
    dsrl_hidden_dims: tuple[int, int, int] = (128, 128, 128)


class SmolVLALeRobotPolicyAdapter(nn.Module, BasePolicy):
    """Inference adapter that serves LeRobot SmolVLA checkpoints in RLinf."""

    def __init__(
        self,
        lerobot_policy: nn.Module,
        preprocessor: Any,
        action_dim: int,
        num_action_chunks: int,
        main_image_key: str = "observation.images.front",
        freeze_vision_encoder: bool | None = None,
        train_expert_only: bool | None = None,
        train_state_proj: bool | None = None,
        use_dsrl: bool = False,
        dsrl_state_dim: int = 14,
        dsrl_action_noise_dim: int = 14,
        dsrl_num_q_heads: int = 2,
        dsrl_image_latent_dim: int = 64,
        dsrl_state_latent_dim: int = 64,
        dsrl_hidden_dims: tuple[int, int, int] = (128, 128, 128),
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.lerobot_policy = lerobot_policy
        self.preprocessor = preprocessor
        self.action_dim = action_dim
        # Keep user-configured output chunk count for RLinf interfaces.
        self._num_action_chunks = num_action_chunks
        # Use checkpoint-native chunk count when constructing DSRL noise,
        # otherwise LeRobot denoiser masks can mismatch (e.g. 50 vs 10).
        self._lerobot_num_action_chunks = self._resolve_lerobot_num_action_chunks(
            lerobot_policy=lerobot_policy,
            fallback=num_action_chunks,
        )
        self.main_image_key = main_image_key
        self._freeze_vision_encoder = freeze_vision_encoder
        self._train_expert_only = train_expert_only
        self._train_state_proj = train_state_proj
        self.use_dsrl = use_dsrl
        self.dsrl_action_noise_dim = dsrl_action_noise_dim
        self.torch_compile_enabled = False
        self.cuda_graph_manager = None

        # Resolve dtype: explicit config > auto-detect from checkpoint > bf16.
        from rlinf.models.embodiment.dtype_utils import resolve_model_dtype

        model_path = getattr(lerobot_policy, "_model_path", None)
        self._model_dtype = resolve_model_dtype(torch_dtype, model_path)

        if self.use_dsrl:
            _dsrl_dtype = self._model_dtype
            dsrl_input_dim = dsrl_state_latent_dim + dsrl_image_latent_dim
            self.dsrl_action_noise_net = GaussianPolicy(
                input_dim=dsrl_input_dim,
                output_dim=dsrl_action_noise_dim,
                hidden_dims=dsrl_hidden_dims,
                low=None,
                high=None,
                action_horizon=1,
            ).to(dtype=_dsrl_dtype)
            self.actor_image_encoder = LightweightImageEncoder64(
                num_images=1,
                latent_dim=dsrl_image_latent_dim,
                image_size=64,
            ).to(dtype=_dsrl_dtype)
            self.actor_state_encoder = CompactStateEncoder(
                state_dim=dsrl_state_dim,
                hidden_dim=dsrl_state_latent_dim,
            ).to(dtype=_dsrl_dtype)
            self.critic_image_encoder = LightweightImageEncoder64(
                num_images=1,
                latent_dim=dsrl_image_latent_dim,
                image_size=64,
            ).to(dtype=_dsrl_dtype)
            self.critic_state_encoder = CompactStateEncoder(
                state_dim=dsrl_state_dim,
                hidden_dim=dsrl_state_latent_dim,
            ).to(dtype=_dsrl_dtype)
            self.q_head = CompactMultiQHead(
                state_dim=dsrl_state_latent_dim,
                image_dim=dsrl_image_latent_dim,
                action_dim=dsrl_action_noise_dim,
                hidden_dims=dsrl_hidden_dims,
                num_q_heads=dsrl_num_q_heads,
                output_dim=1,
            ).to(dtype=_dsrl_dtype)
            
            # Explicitly ensure all DSRL components have requires_grad=True immediately after creation
            # This prevents them from being frozen by subsequent freeze_vlm() calls
            dsrl_components = [
                "dsrl_action_noise_net",
                "actor_image_encoder",
                "actor_state_encoder",
                "critic_image_encoder",
                "critic_state_encoder",
                "q_head"
            ]
            
            # Debug: print all parameter names to understand the model structure
            logger.info("[DSRL init] Listing all parameters to verify model structure:")
            for name, param in self.named_parameters():
                if any(comp in name for comp in dsrl_components):
                    logger.info(f"  Found DSRL param: {name[:100]} -> requires_grad={param.requires_grad}")
            
            # IMPORTANT: Directly unfreeze the DSRL component modules themselves (not just parameters)
            # This ensures the requires_grad flag persists through FSDP wrapping
            if hasattr(self, 'dsrl_action_noise_net'):
                self.dsrl_action_noise_net.requires_grad_(True)
            if hasattr(self, 'actor_image_encoder'):
                self.actor_image_encoder.requires_grad_(True)
            if hasattr(self, 'actor_state_encoder'):
                self.actor_state_encoder.requires_grad_(True)
            if hasattr(self, 'critic_image_encoder'):
                self.critic_image_encoder.requires_grad_(True)
            if hasattr(self, 'critic_state_encoder'):
                self.critic_state_encoder.requires_grad_(True)
            if hasattr(self, 'q_head'):
                self.q_head.requires_grad_(True)
            
            logger.info("[DSRL init] Directly set requires_grad=True on all DSRL component modules")

        if self._lerobot_num_action_chunks != self._num_action_chunks:
            logger.warning(
                "SmolVLA chunk mismatch: cfg num_action_chunks=%d, checkpoint num_action_chunks=%d. "
                "Using checkpoint value for denoising noise shape and cfg value for output truncation.",
                self._num_action_chunks,
                self._lerobot_num_action_chunks,
            )
        if self.use_dsrl:
            # Final verification that critic components are trainable
            logger.info("[DSRL init] Final verification of critic component trainability...")
            critic_components = ["critic_image_encoder", "critic_state_encoder", "q_head"]
            for comp in critic_components:
                if hasattr(self, comp):
                    module = getattr(self, comp)
                    # Double-check requires_grad
                    module.requires_grad_(True)
                    trainable_count = sum(p.requires_grad for p in module.parameters())
                    total_count = sum(1 for _ in module.parameters())
                    logger.info(f"[DSRL init] {comp}: {trainable_count}/{total_count} params trainable")
                    if trainable_count == 0:
                        raise RuntimeError(f"Failed to initialize {comp} with trainable parameters!")

    @staticmethod
    def _resolve_lerobot_num_action_chunks(
        lerobot_policy: nn.Module,
        fallback: int,
    ) -> int:
        cfg = getattr(lerobot_policy, "config", None)
        if cfg is None:
            return fallback

        direct_attrs = (
            "num_action_chunks",
            "n_action_steps",
            "action_horizon",
            "chunk_size",
            "action_chunk_size",
        )
        for attr in direct_attrs:
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                return value

        output_features = getattr(cfg, "output_features", None)
        if isinstance(output_features, dict):
            action_feat = output_features.get("action") or output_features.get("actions")
            if isinstance(action_feat, dict):
                shape = action_feat.get("shape")
            else:
                shape = getattr(action_feat, "shape", None)
            if isinstance(shape, (list, tuple)) and len(shape) > 0:
                first = shape[0]
                if isinstance(first, int) and first > 0:
                    return first

        return fallback

    @property
    def num_action_chunks(self):
        return self._num_action_chunks

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        action_dim: int,
        num_action_chunks: int,
        lerobot_src_path: str | None = None,
        main_image_key: str = "observation.images.front",
        freeze_vision_encoder: bool | None = None,
        train_expert_only: bool | None = None,
        train_state_proj: bool | None = None,
        use_dsrl: bool = False,
        dsrl_state_dim: int = 14,
        dsrl_action_noise_dim: int = 14,
        dsrl_num_q_heads: int = 2,
        dsrl_image_latent_dim: int = 64,
        dsrl_state_latent_dim: int = 64,
        dsrl_hidden_dims: tuple[int, int, int] = (128, 128, 128),
        torch_dtype: torch.dtype | None = None,
    ):
        if lerobot_src_path:
            src = str(Path(lerobot_src_path).expanduser().resolve())
            if src not in sys.path:
                sys.path.insert(0, src)

        make_pre_post_processors, LeRobotSmolVLAPolicy = import_lerobot_policy(
            "lerobot.policies.smolvla.modeling_smolvla",
            "SmolVLAPolicy",
        )

        try:
            # Try standard load which downloads VLM backbone weights.
            lerobot_policy = LeRobotSmolVLAPolicy.from_pretrained(
                model_path,
                local_files_only=True,
            )
        except (OSError, Exception):
            # If the VLM backbone download fails (offline, missing cache),
            # skip it — the finetuned checkpoint's model.safetensors contains
            # all weights including the backbone.
            _orig_init = LeRobotSmolVLAPolicy.__init__

            def _patched_init(self_lr, config, **kw):
                config.load_vlm_weights = False
                _orig_init(self_lr, config, **kw)

            LeRobotSmolVLAPolicy.__init__ = _patched_init
            try:
                lerobot_policy = LeRobotSmolVLAPolicy.from_pretrained(
                    model_path,
                    local_files_only=True,
                )
            finally:
                LeRobotSmolVLAPolicy.__init__ = _orig_init
        try:
            preprocessor, _ = make_pre_post_processors(
                lerobot_policy.config,
                pretrained_path=model_path,
            )
        except TypeError as exc:
            if "pretrained_path" not in str(exc):
                raise
            preprocessor, _ = make_pre_post_processors(lerobot_policy.config)
        lerobot_policy.eval()
        # Tag model_path so __init__ can auto-detect dtype from checkpoint.
        lerobot_policy._model_path = model_path
        return cls(
            lerobot_policy=lerobot_policy,
            preprocessor=preprocessor,
            action_dim=action_dim,
            num_action_chunks=num_action_chunks,
            main_image_key=main_image_key,
            freeze_vision_encoder=freeze_vision_encoder,
            train_expert_only=train_expert_only,
            train_state_proj=train_state_proj,
            use_dsrl=use_dsrl,
            dsrl_state_dim=dsrl_state_dim,
            dsrl_action_noise_dim=dsrl_action_noise_dim,
            dsrl_num_q_heads=dsrl_num_q_heads,
            dsrl_image_latent_dim=dsrl_image_latent_dim,
            dsrl_state_latent_dim=dsrl_state_latent_dim,
            dsrl_hidden_dims=dsrl_hidden_dims,
            torch_dtype=torch_dtype,
        )

    def freeze_vlm(self) -> None:
        """Freeze pretrained weights, following the OpenPI pattern.

        When ``train_expert_only`` is set, this method:
        1. Applies LeRobot-level freeze flags (VLM backbone via
        ``set_requires_grad``).
        2. If ``use_dsrl`` is also enabled, additionally freezes the action
        expert (``lm_expert``) and flow-matching projection layers so
        that only the lightweight DSRL actor/critic modules receive
        gradients.

        This consolidates the previous ``_apply_lerobot_freeze_config`` and
        ``_freeze_for_dsrl`` into a single method, mirroring
        ``OpenPi0ForRLActionPrediction.freeze_vlm()``.
        """
        # ── Step 1: LeRobot-level freeze (VLM backbone) ──────────────
        cfg = getattr(self.lerobot_policy, "config", None)
        if cfg is not None:
            if self._freeze_vision_encoder is not None:
                cfg.freeze_vision_encoder = self._freeze_vision_encoder
            if self._train_expert_only is not None:
                cfg.train_expert_only = self._train_expert_only
            if self._train_state_proj is not None:
                cfg.train_state_proj = self._train_state_proj

        model = getattr(self.lerobot_policy, "model", None)
        if model is not None:
            vlm_with_expert = getattr(model, "vlm_with_expert", None)
            if vlm_with_expert is not None:
                if self._freeze_vision_encoder is not None:
                    vlm_with_expert.freeze_vision_encoder = (
                        self._freeze_vision_encoder
                    )
                if self._train_expert_only is not None:
                    vlm_with_expert.train_expert_only = self._train_expert_only
                if hasattr(vlm_with_expert, "set_requires_grad"):
                    vlm_with_expert.set_requires_grad()
            if hasattr(model, "set_requires_grad"):
                model.set_requires_grad()

        # ── Step 2: DSRL additional freezing ─────────────────────────
        if not self.use_dsrl:
            return

        if model is None:
            logger.warning(
                "[DSRL freeze] lerobot_policy.model not found; skipping."
            )
            return

        # 2a. Freeze the action expert (lm_expert) — analogous to OpenPI
        #     freezing gemma_expert.
        vlm_with_expert = getattr(model, "vlm_with_expert", None)
        if vlm_with_expert is not None:
            lm_expert = getattr(vlm_with_expert, "lm_expert", None)
            if lm_expert is not None:
                lm_expert.eval()
                expert_count = 0
                for p in lm_expert.parameters():
                    p.requires_grad = False
                    expert_count += p.numel()
                logger.info(
                    "[DSRL freeze] Froze lm_expert: %s parameters",
                    f"{expert_count:,}",
                )

        # 2b. Freeze flow-matching projection layers.
        # IMPORTANT: Only freeze projection layers within LeRobot's model,
        # NOT any DSRL components that might have similar names.
        projection_names = [
            "action_in_proj",
            "action_out_proj", 
            "action_time_mlp_in",
            "action_time_mlp_out",
            "state_proj",
        ]
        
        # Define DSRL component names to exclude from freezing
        dsrl_component_names = [
            "dsrl_action_noise_net",
            "actor_image_encoder",
            "actor_state_encoder",
            "critic_image_encoder",
            "critic_state_encoder",
            "q_head"
        ]
        
        frozen_count = 0
        for name, param in model.named_parameters():
            # Skip any DSRL component parameters
            skip = False
            for dsrl_name in dsrl_component_names:
                if dsrl_name in name:
                    skip = True
                    break
            if skip:
                continue
            
            # Check if this is a projection layer we want to freeze
            for proj_name in projection_names:
                # Match either: ".proj_name." or ".proj_name" at end of string
                if f".{proj_name}." in name or name.endswith(f".{proj_name}"):
                    param.requires_grad = False
                    frozen_count += 1
                    logger.debug(f"[DSRL freeze] Froze projection: {name}")
                    break
        
        logger.info(
            "[DSRL freeze] Froze %d projection-layer parameters (%s)",
            frozen_count,
            ", ".join(projection_names),
        )

        # 2c. Explicitly unfreeze DSRL actor/critic components to ensure they are trainable
        dsrl_unfrozen_count = 0
        
        # Debug: Check both self and model parameters to handle LoRA cases
        logger.info("[DSRL unfreeze] Checking parameters in both self and model...")
        
        # First, unfreeze from self (for non-LoRA case)
        debug_count = 0
        total_dsrl_params = 0
        
        # List ALL parameters that contain any DSRL component name
        for name, param in self.named_parameters():
            if any(comp in name for comp in dsrl_component_names):
                total_dsrl_params += 1
                if debug_count < 20:  # Print first 20 matches
                    logger.info(f"[DSRL unfreeze][self] #{debug_count}: {name[:120]} -> requires_grad={param.requires_grad}")
                    debug_count += 1
                
                # Always unfreeze regardless of current state
                if not param.requires_grad:
                    param.requires_grad = True
                    dsrl_unfrozen_count += 1
        
        logger.info(f"[DSRL unfreeze] Total DSRL params found in self: {total_dsrl_params}, unfroze: {dsrl_unfrozen_count}")
        
        # Also check model parameters (for LoRA case where params might be under base_model)
        if hasattr(self, 'lerobot_policy') and hasattr(self.lerobot_policy, 'model'):
            inner_model = self.lerobot_policy.model
            model_debug_count = 0
            model_total_dsrl_params = 0
            
            for name, param in inner_model.named_parameters():
                if any(comp in name for comp in dsrl_component_names):
                    model_total_dsrl_params += 1
                    if model_debug_count < 20:  # Print first 20 matches
                        logger.info(f"[DSRL unfreeze][inner_model] #{model_debug_count}: {name[:120]} -> requires_grad={param.requires_grad}")
                        model_debug_count += 1
                    
                    if not param.requires_grad:
                        param.requires_grad = True
                        dsrl_unfrozen_count += 1
            
            logger.info(f"[DSRL unfreeze] Total DSRL params found in inner_model: {model_total_dsrl_params}, unfroze: {dsrl_unfrozen_count - total_dsrl_params}")
        
        # 2d. Force unfreeze at module level for all DSRL components
        logger.info("[DSRL unfreeze] Force unfreezing DSRL component modules...")
        for comp_name in dsrl_component_names:
            # Check if component exists in self
            if hasattr(self, comp_name):
                module = getattr(self, comp_name)
                # Use requires_grad_() on the module to ensure all submodules are unfrozen
                module.requires_grad_(True)
                logger.info(f"[DSRL unfreeze] Module {comp_name} forced to requires_grad=True")
            
            # Also check in lerobot_policy.model for LoRA case
            if hasattr(self, 'lerobot_policy') and hasattr(self.lerobot_policy, 'model'):
                if hasattr(self.lerobot_policy.model, comp_name):
                    module = getattr(self.lerobot_policy.model, comp_name)
                    module.requires_grad_(True)
                    logger.info(f"[DSRL unfreeze] Module {comp_name} in inner_model forced to requires_grad=True")
        
        # 2e. Log final trainable parameter summary
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        total = sum(p.numel() for p in self.parameters())
        
        logger.info(
            "[DSRL freeze] Froze lm_expert and projection layers, unfroze %d DSRL component parameters",
            dsrl_unfrozen_count,
        )
        logger.info(
            "[DSRL freeze] Trainable: %s / %s (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / total if total else 0,
        )
        
        # 2f. Verify critic components are trainable
        logger.info("[DSRL freeze] Verifying critic component trainability...")
        critic_components = ["critic_image_encoder", "critic_state_encoder", "q_head"]
        for comp in critic_components:
            trainable_params = 0
            total_params = 0
            for name, param in self.named_parameters():
                if comp in name:
                    total_params += 1
                    if param.requires_grad:
                        trainable_params += 1
            logger.info(f"[DSRL freeze] {comp}: {trainable_params}/{total_params} parameters trainable")
            if trainable_params == 0 and total_params > 0:
                logger.error(f"[DSRL freeze] CRITICAL: {comp} has no trainable parameters! This will cause optimizer creation to fail.")
        
        # 2g. IMPORTANT: Also verify at module level (handles FSDP wrapped cases)
        logger.info("[DSRL freeze] Module-level requires_grad verification:")
        for comp_name in dsrl_component_names:
            if hasattr(self, comp_name):
                module = getattr(self, comp_name)
                module_requires_grad = next(module.parameters()).requires_grad
                logger.info(f"[DSRL freeze] Module {comp_name}: requires_grad={module_requires_grad}")

    def _to_lerobot_batch(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        states = env_obs["states"]
        if not torch.is_tensor(states):
            states = torch.as_tensor(states)

        images = env_obs["main_images"]
        if not torch.is_tensor(images):
            images = torch.as_tensor(images)
        if images.ndim != 4:
            raise ValueError(f"Expected main_images shape [B,H,W,C], got {tuple(images.shape)}")
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        if images.max() > 1.0:
            images = images.float() / 255.0

        tasks = env_obs.get("task_descriptions")
        if tasks is None:
            tasks = [""] * int(states.shape[0])

        return {
            "observation.state": states,
            self.main_image_key: images,
            "task": tasks,
        }

    def _preprocess_dsrl_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                f"Expected image shape [B,C,H,W] or [B,H,W,C], got {tuple(images.shape)}"
            )
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.float()
            if images.min() < 0:
                images = (images + 1.0) / 2.0
        images = images.clamp(0.0, 1.0)
        images = F.interpolate(images, size=(64, 64), mode="bilinear", align_corners=False)
        images = images * 2.0 - 1.0
        return images.unsqueeze(1)

    def _preprocess_states(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() > 2:
            states = states.reshape(states.shape[0], -1)
        if states.dtype != self._model_dtype:
            states = states.to(self._model_dtype)
        return states

    def _get_dsrl_images(self, obs: dict) -> torch.Tensor:
        """Extract images for DSRL encoders from obs dict.

        Handles ``main_images`` key, nested ``images`` dict, and flat
        image keys at the top level (e.g. ``obs[main_image_key]``).
        """
        raw = obs.get("main_images")
        if raw is None:
            images_dict = obs.get("images", {})
            if isinstance(images_dict, dict):
                raw = images_dict.get(self.main_image_key)
        if raw is None:
            raw = obs.get(self.main_image_key)
        if raw is None:
            raise KeyError(
                f"SmolVLALeRobotPolicyAdapter DSRL: cannot find images in obs. "
                f"Available keys: {list(obs.keys())}"
            )
        return raw

    def sac_forward(self, obs=None, data=None, train=False, **kwargs):
        if not self.use_dsrl:
            raise ValueError("sac_forward called but SmolVLA use_dsrl=False")
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})
        images = self._preprocess_dsrl_images(self._get_dsrl_images(obs))
        states = self._preprocess_states(obs["states"])
        device = next(self.actor_image_encoder.parameters()).device
        images = images.to(device=device, dtype=self._model_dtype)
        states = states.to(device=device, dtype=self._model_dtype)
        image_features = self.actor_image_encoder(images)
        state_features = self.actor_state_encoder(states)
        features = torch.cat([state_features, image_features], dim=-1)
        deterministic = kwargs.get("mode", "train") == "eval"
        action_noise, logprobs = self.dsrl_action_noise_net.sample(
            features, deterministic=deterministic
        )
        return action_noise, logprobs, None

    def sac_q_forward(
        self,
        obs=None,
        data=None,
        actions=None,
        detach_encoder=False,
        train=False,
        **kwargs,
    ):
        if not self.use_dsrl:
            raise ValueError("sac_q_forward called but SmolVLA use_dsrl=False")
        if obs is None:
            obs = data.get("obs", data) if data is not None else kwargs.get("obs", {})
        if actions is None:
            actions = kwargs.get("actions")
        images = self._preprocess_dsrl_images(self._get_dsrl_images(obs))
        states = self._preprocess_states(obs["states"])
        device = next(self.critic_image_encoder.parameters()).device
        images = images.to(device=device, dtype=self._model_dtype)
        states = states.to(device=device, dtype=self._model_dtype)
        actions = actions.to(device=device, dtype=self._model_dtype)
        image_features = self.critic_image_encoder(images)
        state_features = self.critic_state_encoder(states)
        if detach_encoder:
            image_features = image_features.detach()
            state_features = state_features.detach()
        if actions.dim() == 3:
            actions = actions[:, 0, :]
        return self.q_head(state_features, image_features, actions)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        raise NotImplementedError(
            f"{forward_type=} is not supported in LeRobot SmolVLA adapter mode."
        )

    def default_forward(
        self,
        forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        **kwargs,
    ):
        raise NotImplementedError(
            "LeRobot SmolVLA checkpoint mode currently supports rollout inference only; "
            "training-time default_forward is not implemented yet."
        )

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        batch = self._to_lerobot_batch(env_obs)
        processed = self.preprocessor(batch)
        # Ensure all image tensors are float — LeRobot's resize_with_pad
        # uses F.interpolate which doesn't support uint8.
        for k, v in processed.items():
            if torch.is_tensor(v) and v.dtype == torch.uint8:
                processed[k] = v.float() / 255.0
        if self.use_dsrl and mode == "train":
            noise_actions, noise_logprob, _ = self.sac_forward(
                obs=env_obs, train=False, mode=mode
            )
            # noise_actions: (B, 1, dsrl_action_noise_dim) from GaussianPolicy.
            # Flatten to (B, dsrl_action_noise_dim) then broadcast across chunks.
            # Use dsrl_action_noise_dim (== max_action_dim == 32) so the full noise
            # vector fills the denoiser's action_in_proj.  Slicing to action_dim
            # (== 16) would zero-pad half the input and waste SAC's learned signal.
            B = noise_actions.shape[0]
            noise_flat = noise_actions.reshape(B, -1)  # (B, dsrl_action_noise_dim)
            max_action_dim = self.lerobot_policy.config.max_action_dim
            # Take up to max_action_dim noise dims; pad only if dsrl_action_noise_dim
            # is smaller than max_action_dim (config mismatch).
            noise_per_step = noise_flat[:, :max_action_dim]  # (B, max_action_dim)
            if noise_per_step.shape[-1] < max_action_dim:
                noise_per_step = F.pad(
                    noise_per_step, (0, max_action_dim - noise_per_step.shape[-1])
                )
            policy_noise = noise_per_step.unsqueeze(1).expand(
                B, self._lerobot_num_action_chunks, -1
            ).contiguous()
            # Use autocast to handle dtype mismatches between model weights
            # and LeRobot's internal float32 noise generation.
            device_type = "cuda" if next(self.lerobot_policy.parameters()).is_cuda else "cpu"
            with torch.autocast(device_type=device_type, dtype=self._model_dtype):
                actions = self.lerobot_policy.predict_action_chunk(processed, noise=policy_noise)
            prev_logprobs = noise_logprob
            # Flatten noise_actions to 2D [B, action_horizon * noise_dim] for
            # consistency with the embodied IO struct that expects 2D actions.
            forward_action = noise_actions.reshape(B, -1)
        else:
            device_type = "cuda" if next(self.lerobot_policy.parameters()).is_cuda else "cpu"
            with torch.autocast(device_type=device_type, dtype=self._model_dtype):
                actions = self.lerobot_policy.predict_action_chunk(processed)
            prev_logprobs = torch.zeros(
                (actions.shape[0],), device=actions.device, dtype=actions.dtype
            )
            forward_action = actions.reshape(actions.shape[0], -1)
        output_chunks = min(self._num_action_chunks, int(actions.shape[1]))
        actions = actions[:, :output_chunks, : self.action_dim]

        chunk_actions = actions.detach().cpu().float().numpy()
        prev_values = torch.zeros(
            (actions.shape[0], 1), device=actions.device, dtype=actions.dtype
        )

        forward_inputs = {"action": forward_action}
        if return_obs:
            forward_inputs["states"] = batch["observation.state"]
            forward_inputs["main_images"] = self._get_dsrl_images(env_obs) if self.use_dsrl else env_obs.get("main_images")

        result = {
            "prev_logprobs": prev_logprobs if calculate_logprobs else None,
            "prev_values": prev_values if calculate_values else None,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result


class SmolVLAPolicy(nn.Module, BasePolicy):
    def __init__(self, cfg: SmolVLAConfig):
        super().__init__()
        self.cfg = cfg
        self.torch_compile_enabled = False
        self.cuda_graph_manager = None

        channels = int(cfg.image_size[0]) if len(cfg.image_size) == 3 else 3
        self.image_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, cfg.hidden_dim),
            nn.GELU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
        )
        self.output_dim = cfg.action_dim * cfg.num_action_chunks
        self.actor_mean = nn.Linear(cfg.hidden_dim, self.output_dim)
        self.actor_logstd = nn.Linear(cfg.hidden_dim, self.output_dim)
        if cfg.add_value_head:
            self.value_head = ValueHead(
                input_dim=cfg.hidden_dim,
                hidden_sizes=(cfg.hidden_dim, cfg.hidden_dim),
                activation="relu",
            )

    @property
    def num_action_chunks(self):
        return self.cfg.num_action_chunks

    def preprocess_env_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        states = env_obs["states"].to(device=device).float()

        if "main_images" not in env_obs:
            raise KeyError("SmolVLAPolicy requires `main_images` in env observations.")
        main_images = env_obs["main_images"].to(device=device)
        if main_images.ndim == 4 and main_images.shape[-1] == 3:
            main_images = main_images.permute(0, 3, 1, 2)
        main_images = main_images.float()
        if main_images.max() > 1.0:
            main_images = main_images / 255.0
        main_images = main_images * 2.0 - 1.0

        return {"states": states, "main_images": main_images}

    def _encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        image_feat = self.image_encoder(obs["main_images"])
        state_feat = self.state_encoder(obs["states"])
        return self.fusion(torch.cat([image_feat, state_feat], dim=-1))

    def _dist(self, fused: torch.Tensor) -> Normal:
        mean = self.actor_mean(fused)
        logstd = self.actor_logstd(fused).clamp(
            min=float(torch.log(torch.tensor(self.cfg.min_std))),
            max=float(torch.log(torch.tensor(self.cfg.max_std))),
        )
        return Normal(mean, torch.exp(logstd))

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"{forward_type=} is not supported for SmolVLAPolicy.")

    def default_forward(
        self,
        forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        **kwargs,
    ):
        obs = {
            "states": forward_inputs["states"],
            "main_images": forward_inputs["main_images"],
        }
        obs = self.preprocess_env_obs(obs)
        actions = forward_inputs["action"]
        if actions.ndim == 3:
            actions = actions.reshape(actions.shape[0], -1)

        fused = self._encode(obs)
        probs = self._dist(fused)
        output_dict = {}
        if compute_logprobs:
            logprobs = probs.log_prob(actions).reshape(
                -1, self.cfg.num_action_chunks, self.cfg.action_dim
            )
            output_dict["logprobs"] = logprobs
        if compute_entropy:
            entropy = probs.entropy().reshape(
                -1, self.cfg.num_action_chunks, self.cfg.action_dim
            )
            output_dict["entropy"] = entropy
        if compute_values:
            if hasattr(self, "value_head"):
                output_dict["values"] = self.value_head(fused)
            else:
                raise NotImplementedError("Value head is disabled for this SmolVLA config.")
        return output_dict

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        obs = self.preprocess_env_obs(env_obs)
        fused = self._encode(obs)
        probs = self._dist(fused)

        if mode == "eval":
            actions = probs.mean
        else:
            actions = probs.sample()

        chunk_actions = actions.reshape(-1, self.cfg.num_action_chunks, self.cfg.action_dim)
        chunk_logprobs = probs.log_prob(actions).reshape(
            -1, self.cfg.num_action_chunks, self.cfg.action_dim
        )

        if hasattr(self, "value_head") and calculate_values:
            values = self.value_head(fused)
        else:
            values = torch.zeros(
                (actions.shape[0], 1), device=actions.device, dtype=actions.dtype
            )

        forward_inputs = {"action": chunk_actions}
        if return_obs:
            forward_inputs["states"] = obs["states"]
            forward_inputs["main_images"] = obs["main_images"]

        result = {
            "prev_logprobs": chunk_logprobs if calculate_logprobs else None,
            "prev_values": values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions.detach().cpu().numpy(), result
