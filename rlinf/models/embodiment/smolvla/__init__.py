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

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from rlinf.models.embodiment.smolvla.smolvla_policy import (
        SmolVLAConfig,
        SmolVLALeRobotPolicyAdapter,
        SmolVLAPolicy,
    )

    if cfg.get("use_lerobot_checkpoint", False):
        model = SmolVLALeRobotPolicyAdapter.from_pretrained(
            model_path=cfg.model_path,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            lerobot_src_path=cfg.get("lerobot_src_path", None),
            main_image_key=cfg.get("main_image_key", "observation.images.front"),
            freeze_vision_encoder=cfg.get("freeze_vision_encoder", None),
            train_expert_only=cfg.get("train_expert_only", None),
            train_state_proj=cfg.get("train_state_proj", None),
            use_dsrl=cfg.get("use_dsrl", False),
            dsrl_state_dim=cfg.get("dsrl_state_dim", cfg.get("state_dim", cfg.action_dim)),
            dsrl_action_noise_dim=cfg.get(
                "dsrl_action_noise_dim", cfg.action_dim * cfg.num_action_chunks
            ),
            dsrl_num_q_heads=cfg.get("dsrl_num_q_heads", 2),
            dsrl_image_latent_dim=cfg.get("dsrl_image_latent_dim", 64),
            dsrl_state_latent_dim=cfg.get("dsrl_state_latent_dim", 64),
            dsrl_hidden_dims=tuple(cfg.get("dsrl_hidden_dims", [128, 128, 128])),
            torch_dtype=torch_dtype,
        )
        # Freeze pretrained weights after DSRL modules are created,
        # following the OpenPI pattern (openpi/__init__.py:61-62).
        if cfg.get("train_expert_only", False):
            model.freeze_vlm()
        return model

    model_cfg = SmolVLAConfig(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        hidden_dim=cfg.get("hidden_dim", 512),
        image_size=cfg.get("image_size", [3, 224, 224]),
        add_value_head=cfg.get("add_value_head", False),
        min_std=cfg.get("min_std", 1e-4),
        max_std=cfg.get("max_std", 10.0),
        lerobot_src_path=cfg.get("lerobot_src_path", None),
        use_dsrl=cfg.get("use_dsrl", False),
        dsrl_state_dim=cfg.get("dsrl_state_dim", cfg.get("state_dim", cfg.action_dim)),
        dsrl_action_noise_dim=cfg.get(
            "dsrl_action_noise_dim", cfg.action_dim * cfg.num_action_chunks
        ),
        dsrl_num_q_heads=cfg.get("dsrl_num_q_heads", 2),
        dsrl_image_latent_dim=cfg.get("dsrl_image_latent_dim", 64),
        dsrl_state_latent_dim=cfg.get("dsrl_state_latent_dim", 64),
        dsrl_hidden_dims=tuple(cfg.get("dsrl_hidden_dims", [128, 128, 128])),
    )
    return SmolVLAPolicy(model_cfg)
