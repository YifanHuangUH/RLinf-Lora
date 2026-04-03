# Copyright 2025 The RLinf Authors.
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
from importlib import import_module
from typing import Protocol

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.scheduler import RosRobotHWInfo, WorkerInfo


class RosRobotAdapter(Protocol):
    """Interface for user-provided ROS robot adapters."""

    def reset(self) -> tuple[dict, dict]: ...

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]: ...


@dataclass
class RosRobotConfig:
    is_dummy: bool = True
    max_num_steps: int = 100
    step_frequency: float = 10.0
    action_scale: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.05, 1.0]))
    state_dim: int = 14
    action_dim: int = 16
    camera_keys: list[str] = field(default_factory=lambda: ["observation.images.front"])
    image_shape: tuple[int, int, int] = (128, 128, 3)
    adapter_cls: str = "rlinf.envs.realworld.ros_robot.ros2_jointstate_adapter.Ros2JointStateAdapter"
    state_topic: str = "/io_teleop/joint_states"
    command_topic: str = "/policy/joint_cmd"
    gripper_command_topic: str = "/policy/target_gripper_status"
    joint_names: list[str] = field(
        default_factory=lambda: [
            "right_joint_1.pos",
            "right_joint_2.pos",
            "right_joint_3.pos",
            "right_joint_4.pos",
            "right_joint_5.pos",
            "right_joint_6.pos",
            "right_joint_7.pos",
            "left_joint_1.pos",
            "left_joint_2.pos",
            "left_joint_3.pos",
            "left_joint_4.pos",
            "left_joint_5.pos",
            "left_joint_6.pos",
            "left_joint_7.pos",
        ]
    )
    image_topics: dict[str, str] = field(default_factory=dict)
    qos_history_depth: int = 10


class RosRobotEnv(gym.Env):
    """ROS-backed robot environment scaffold for RealWorldEnv integration."""

    def __init__(
        self,
        config: RosRobotConfig,
        worker_info: WorkerInfo | None,
        hardware_info: RosRobotHWInfo | None,
        env_idx: int,
    ):
        self.config = config
        self.worker_info = worker_info
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self._num_steps = 0
        self._state = np.zeros((self.config.state_dim,), dtype=np.float32)
        self._adapter = None
        if not self.config.is_dummy:
            self._adapter = self._build_adapter()

        self.action_space = gym.spaces.Box(
            low=-np.ones((self.config.action_dim,), dtype=np.float32),
            high=np.ones((self.config.action_dim,), dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "state": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(self.config.state_dim,),
                            dtype=np.float32,
                        )
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            0, 255, shape=self.config.image_shape, dtype=np.uint8
                        )
                        for key in self.config.camera_keys
                    }
                ),
            }
        )

    def _build_adapter(self):
        if not self.config.adapter_cls:
            raise ValueError(
                "RosRobotConfig.adapter_cls must be provided when is_dummy is False."
            )
        module_name, cls_name = self.config.adapter_cls.rsplit(".", 1)
        adapter_cls = getattr(import_module(module_name), cls_name)
        return adapter_cls(
            config=self.config,
            worker_info=self.worker_info,
            hardware_info=self.hardware_info,
        )

    def _dummy_observation(self) -> dict:
        frames = {
            key: np.zeros(self.config.image_shape, dtype=np.uint8)
            for key in self.config.camera_keys
        }
        return {"state": {"state": self._state.copy()}, "frames": frames}

    def reset(self, *, seed=None, options=None):
        self._num_steps = 0
        if self.config.is_dummy:
            return self._dummy_observation(), {}
        return self._adapter.reset()

    def step(self, action: np.ndarray):
        if not self.config.is_dummy:
            return self._adapter.step(action)

        action = np.clip(action, self.action_space.low, self.action_space.high)
        copy_len = min(self.config.state_dim, self.config.action_dim)
        self._state[:copy_len] = action[:copy_len]
        if self.config.state_dim >= 7 and self.config.action_dim >= 6:
            pose = np.zeros((7,), dtype=np.float32)
            pose[:3] = action[:3] * self.config.action_scale[0]
            pose[3:] = R.from_euler("xyz", action[3:6] * self.config.action_scale[1]).as_quat()
            self._state[:7] = pose
        self._num_steps += 1
        observation = self._dummy_observation()
        terminated = False
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, 0.0, terminated, truncated, {}

    @property
    def task_description(self):
        return "Control a ROS robot with end-effector actions."
