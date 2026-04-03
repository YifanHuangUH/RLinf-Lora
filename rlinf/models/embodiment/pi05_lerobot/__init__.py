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
    from rlinf.models.embodiment.pi05_lerobot.pi05_lerobot_policy import (
        Pi05LeRobotPolicyAdapter,
    )

    image_keys = list(cfg.get("image_keys", ["observation.images.front"]))

    return Pi05LeRobotPolicyAdapter.from_pretrained(
        model_path=cfg.model_path,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        image_keys=image_keys,
        freeze_vision_encoder=cfg.get("freeze_vision_encoder", None),
        train_expert_only=cfg.get("train_expert_only", None),
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
