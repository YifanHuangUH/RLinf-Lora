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
from typing import Optional

from ..hardware import (
    Hardware,
    HardwareConfig,
    HardwareInfo,
    HardwareResource,
    NodeHardwareConfig,
)


@dataclass
class RosRobotHWInfo(HardwareInfo):
    """Hardware information for ROS robot resources."""

    config: "RosRobotConfig"


@Hardware.register()
class RosRobot(Hardware):
    """Hardware policy for generic ROS-controlled robots."""

    HW_TYPE = "RosRobot"

    @classmethod
    def enumerate(
        cls, node_rank: int, configs: Optional[list["RosRobotConfig"]] = None
    ) -> Optional[HardwareResource]:
        assert configs is not None, "RosRobot hardware requires explicit configurations"
        robot_configs = [
            config
            for config in configs
            if isinstance(config, RosRobotConfig) and config.node_rank == node_rank
        ]
        if not robot_configs:
            return None
        infos = [
            RosRobotHWInfo(type=cls.HW_TYPE, model=config.robot_model, config=config)
            for config in robot_configs
        ]
        return HardwareResource(type=cls.HW_TYPE, infos=infos)


@NodeHardwareConfig.register_hardware_config(RosRobot.HW_TYPE)
@dataclass
class RosRobotConfig(HardwareConfig):
    """Configuration for generic ROS robot resources."""

    robot_model: str = "ros_robot"
    namespace: str = ""

    def __post_init__(self):
        assert isinstance(self.node_rank, int), (
            f"'node_rank' in RosRobot config must be an integer. But got {type(self.node_rank)}."
        )
