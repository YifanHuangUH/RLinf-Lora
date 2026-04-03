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

from dataclasses import dataclass

from rlinf.envs.realworld.ros_robot.ros_robot_env import RosRobotConfig, RosRobotEnv


@dataclass
class RosRobotTaskConfig(RosRobotConfig):
    pass


class RosRobotTaskEnv(RosRobotEnv):
    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        config = RosRobotTaskConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)
