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

"""LeRobot-based Pi0.5 adapter for RLinf.

Loads Pi0.5 checkpoints trained with LeRobot (HuggingFace transformers
architecture) and wraps them behind RLinf's BasePolicy interface with
optional DSRL (SAC / SAC-Q) support.  Mirrors the pattern established
by the SmolVLA LeRobot adapter.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.lerobot_import_utils import import_lerobot_policy
from rlinf.models.embodiment.modules.compact_encoders import (
    CompactMultiQHead,
    CompactStateEncoder,
    LightweightImageEncoder64,
)
from rlinf.models.embodiment.modules.gaussian_policy import GaussianPolicy


class Pi05LeRobotPolicyAdapter(nn.Module, BasePolicy):
    """Inference adapter that serves LeRobot Pi0.5 checkpoints in RLinf."""

    def __init__(
        self,
        lerobot_policy: nn.Module,
        preprocessor: Any,
        action_dim: int,
        num_action_chunks: int,
        image_keys: list[str] | None = None,
        freeze_vision_encoder: bool | None = None,
        train_expert_only: bool | None = None,
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
        self._num_action_chunks = num_action_chunks
        self.image_keys = image_keys or ["observation.images.front"]
        self._apply_lerobot_freeze_config(
            freeze_vision_encoder=freeze_vision_encoder,
            train_expert_only=train_expert_only,
        )
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

    @property
    def num_action_chunks(self):
        return self._num_action_chunks

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        action_dim: int,
        num_action_chunks: int,
        image_keys: list[str] | None = None,
        freeze_vision_encoder: bool | None = None,
        train_expert_only: bool | None = None,
        use_dsrl: bool = False,
        dsrl_state_dim: int = 14,
        dsrl_action_noise_dim: int = 14,
        dsrl_num_q_heads: int = 2,
        dsrl_image_latent_dim: int = 64,
        dsrl_state_latent_dim: int = 64,
        dsrl_hidden_dims: tuple[int, int, int] = (128, 128, 128),
        torch_dtype: torch.dtype | None = None,
    ):
        make_pre_post_processors, LeRobotPI05Policy = import_lerobot_policy(
            "lerobot.policies.pi05.modeling_pi05",
            "PI05Policy",
        )

        lerobot_policy = LeRobotPI05Policy.from_pretrained(
            model_path,
            local_files_only=True,
        )
        try:
            preprocessor, _ = make_pre_post_processors(
                lerobot_policy.config,
                pretrained_path=model_path,
            )
        except TypeError as exc:
            if "pretrained_path" not in str(exc):
                raise
            preprocessor, _ = make_pre_post_processors(lerobot_policy.config)
        except (OSError, ValueError):
            # Preprocessor creation may fail if the tokenizer requires
            # HuggingFace auth (gated paligemma model).  Fall back to the
            # policy's own saved preprocessor files when available.
            preprocessor = _load_saved_preprocessor(model_path)

        lerobot_policy.eval()
        # Tag model_path so __init__ can auto-detect dtype from checkpoint.
        lerobot_policy._model_path = model_path
        return cls(
            lerobot_policy=lerobot_policy,
            preprocessor=preprocessor,
            action_dim=action_dim,
            num_action_chunks=num_action_chunks,
            image_keys=image_keys,
            freeze_vision_encoder=freeze_vision_encoder,
            train_expert_only=train_expert_only,
            use_dsrl=use_dsrl,
            dsrl_state_dim=dsrl_state_dim,
            dsrl_action_noise_dim=dsrl_action_noise_dim,
            dsrl_num_q_heads=dsrl_num_q_heads,
            dsrl_image_latent_dim=dsrl_image_latent_dim,
            dsrl_state_latent_dim=dsrl_state_latent_dim,
            dsrl_hidden_dims=dsrl_hidden_dims,
            torch_dtype=torch_dtype,
        )

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def _apply_lerobot_freeze_config(
        self,
        freeze_vision_encoder: bool | None,
        train_expert_only: bool | None,
    ) -> None:
        cfg = getattr(self.lerobot_policy, "config", None)
        if cfg is not None:
            if freeze_vision_encoder is not None:
                cfg.freeze_vision_encoder = freeze_vision_encoder
            if train_expert_only is not None:
                cfg.train_expert_only = train_expert_only

        model = getattr(self.lerobot_policy, "model", None)
        if model is not None:
            pwe = getattr(model, "paligemma_with_expert", None)
            if pwe is not None:
                if freeze_vision_encoder is not None:
                    pwe.freeze_vision_encoder = freeze_vision_encoder
                if train_expert_only is not None:
                    pwe.train_expert_only = train_expert_only
                if hasattr(pwe, "set_requires_grad"):
                    pwe.set_requires_grad()
            if hasattr(model, "set_requires_grad"):
                model.set_requires_grad()

    # ------------------------------------------------------------------
    # Observation → LeRobot batch conversion
    # ------------------------------------------------------------------

    def _to_lerobot_batch(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """Convert RLinf env observations to a LeRobot-compatible batch dict.

        Expected *env_obs* keys:
        - ``"states"``: [B, state_dim]
        - ``"main_images"`` or per-camera keys matching ``self.image_keys``
          (e.g. ``"observation.images.front"``): [B, H, W, C] uint8/float
        - ``"task_descriptions"`` (optional): list[str]
        """
        states = env_obs["states"]
        if not torch.is_tensor(states):
            states = torch.as_tensor(states)

        batch: dict[str, Any] = {"observation.state": states}

        # Populate each expected image key
        for key in self.image_keys:
            raw = env_obs.get(key)
            if raw is None:
                # Check nested "images" dict (env worker format)
                images_dict = env_obs.get("images", {})
                if isinstance(images_dict, dict):
                    raw = images_dict.get(key)
            if raw is None:
                # Fall back to positional main_images tensor for single-camera
                raw = env_obs.get("main_images")
            if raw is None:
                raise KeyError(
                    f"Pi05LeRobotPolicyAdapter: missing image key '{key}' in env_obs. "
                    f"Available: {list(env_obs.keys())}"
                )
            images = raw if torch.is_tensor(raw) else torch.as_tensor(raw)
            if images.ndim != 4:
                raise ValueError(
                    f"Expected image shape [B,H,W,C] or [B,C,H,W], got {tuple(images.shape)}"
                )
            if images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)
            if images.max() > 1.0:
                images = images.float() / 255.0
            batch[key] = images

        tasks = env_obs.get("task_descriptions")
        if tasks is None:
            tasks = [""] * int(states.shape[0])
        batch["task"] = tasks

        return batch

    # ------------------------------------------------------------------
    # DSRL helpers (shared with SmolVLA adapter)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Forward dispatch
    # ------------------------------------------------------------------

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        raise NotImplementedError(
            f"{forward_type=} is not supported in LeRobot Pi0.5 adapter mode."
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
            "LeRobot Pi0.5 checkpoint mode currently supports rollout inference only; "
            "training-time default_forward is not implemented yet."
        )

    # ------------------------------------------------------------------
    # SAC / DSRL
    # ------------------------------------------------------------------

    def _get_dsrl_images(self, obs: dict) -> torch.Tensor:
        """Extract images for DSRL encoders from obs dict.

        Handles ``main_images`` key, nested ``images`` dict, and flat
        image keys at the top level (e.g. ``obs["observation.images.front"]``).
        """
        raw = obs.get("main_images")
        if raw is None:
            images_dict = obs.get("images", {})
            if isinstance(images_dict, dict) and self.image_keys:
                raw = images_dict.get(self.image_keys[0])
        if raw is None and self.image_keys:
            raw = obs.get(self.image_keys[0])
        if raw is None:
            raise KeyError(
                f"Pi05LeRobotPolicyAdapter DSRL: cannot find images in obs. "
                f"Available keys: {list(obs.keys())}"
            )
        return raw

    def sac_forward(self, obs=None, data=None, train=False, **kwargs):
        if not self.use_dsrl:
            raise ValueError("sac_forward called but Pi05 use_dsrl=False")
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
            raise ValueError("sac_q_forward called but Pi05 use_dsrl=False")
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

    # ------------------------------------------------------------------
    # Rollout inference
    # ------------------------------------------------------------------

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
        # NOTE: This float32 cast is required regardless of model dtype.
        # The LeRobot preprocessor calls .cpu().numpy() which does not
        # support bfloat16.  This is a numpy API limitation, not related
        # to the model's training precision.
        pp_batch = {}
        for k, v in batch.items():
            if torch.is_tensor(v) and v.is_floating_point():
                pp_batch[k] = v.float()
            else:
                pp_batch[k] = v
        processed = self.preprocessor(pp_batch) if self.preprocessor is not None else batch

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
                B, self._num_action_chunks, -1
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
        actions = actions[:, : self._num_action_chunks, : self.action_dim]

        chunk_actions = actions.detach().cpu().float().numpy()
        prev_values = torch.zeros(
            (actions.shape[0], 1), device=actions.device, dtype=actions.dtype
        )

        forward_inputs = {"action": forward_action}
        if return_obs:
            forward_inputs["states"] = batch["observation.state"]
            main_img = env_obs.get("main_images")
            if main_img is None:
                main_img = env_obs.get(self.image_keys[0])
            if main_img is None:
                images_dict = env_obs.get("images", {})
                if isinstance(images_dict, dict):
                    main_img = images_dict.get(self.image_keys[0])
            forward_inputs["main_images"] = main_img

        result = {
            "prev_logprobs": prev_logprobs if calculate_logprobs else None,
            "prev_values": prev_values if calculate_values else None,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result


# ------------------------------------------------------------------
# Utility: fallback preprocessor loading
# ------------------------------------------------------------------

def _load_saved_preprocessor(model_path: str):
    """Try to load the preprocessor pipeline from saved JSON files
    in the checkpoint directory.  Returns ``None`` if unsuccessful."""
    import os

    pp_json = os.path.join(model_path, "policy_preprocessor.json")
    if not os.path.exists(pp_json):
        return None

    try:
        from lerobot.processor.pipeline import PolicyProcessorPipeline

        return PolicyProcessorPipeline.from_pretrained(
            model_path,
            config_filename="policy_preprocessor.json",
        )
    except Exception:
        return None
