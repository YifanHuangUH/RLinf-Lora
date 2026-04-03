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
import sys
import select
import tty
import termios
import time

from dataclasses import dataclass, field
from importlib import import_module
from typing import Optional, Protocol

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

from rlinf.scheduler import RosRobotHWInfo, WorkerInfo


class RosRobotAdapter(Protocol):
    """Interface for user-provided ROS robot adapters."""

    def reset(self) -> tuple[dict, dict]: ...

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]: ...


@dataclass
class RosRobotConfig:
    is_dummy: bool = True
    max_num_steps: int = 50
    max_episode_steps: int | None = None  # Alias for max_num_steps (for backward compatibility)
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
    # 奖励相关配置
    reward_mode: str = "dsrl_terminal"  # "raw", "only_success", "dense", "sparse", "dsrl_terminal"
    use_rel_reward: bool = False
    success_reward: float = 1.0
    failure_penalty: float = -1.0
    step_penalty: float = -0.01
    keyboard_reward_wrapper: Optional[str] = None  # Changed from "single_stage" to None
    use_terminal_reward_wrapper: bool = False  # Enable terminal-based human reward input
    dsrl_step_reward: float = -1.0
    dsrl_success_terminal_reward: float = 0.0
    dsrl_failure_terminal_reward: float = -1.0
    
    # 自动评估配置（用于训练阶段，避免人工阻塞）
    auto_evaluate: bool | str = True       # True: 始终自动，False/null: 始终人工，'after_warmup': 预热后自动
    warmup_steps: int = 10                 # 前 N 个 step 使用人工标注（Replay Buffer 积累期）
    success_threshold: float = -10.0       # return > -10 视为成功


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

        # 奖励相关属性
        self.use_rel_reward = getattr(config, 'use_rel_reward', False)
        self.reward_mode = getattr(config, 'reward_mode', 'dsrl_terminal')
        self.success_reward = getattr(config, 'success_reward', 1.0)
        self.failure_penalty = getattr(config, 'failure_penalty', -1.0)
        self.step_penalty = getattr(config, 'step_penalty', -0.01)
        self.keyboard_reward_wrapper = getattr(config, 'keyboard_reward_wrapper', None)  # Changed default to None
        self.dsrl_step_reward = getattr(config, 'dsrl_step_reward', -1.0)
        self.dsrl_success_terminal_reward = getattr(config, 'dsrl_success_terminal_reward', 0.0)
        self.dsrl_failure_terminal_reward = getattr(config, 'dsrl_failure_terminal_reward', -1.0)
        
        # 自动评估配置（优先级高于 keyboard_reward_wrapper）
        self.auto_evaluate = getattr(config, 'auto_evaluate', False)
        self.warmup_steps = getattr(config, 'warmup_steps', 10)
        self.success_threshold = getattr(config, 'success_threshold', -10.0)
        
        # 初始化日志记录器
        from rlinf.utils.logging import get_logger
        self.logger = get_logger()
        
        self.prev_step_reward = torch.zeros(1, dtype=torch.float32)  # [B, ]
        self._episode_evaluated = False
        
        # 添加成功/失败标志
        self.success = False
        self.failure = False
        
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

    def _get_max_steps(self) -> int:
        """获取最大步数配置，支持 max_num_steps 和 max_episode_steps 两种字段名"""
        # 优先使用 max_num_steps
        if hasattr(self.config, 'max_num_steps') and self.config.max_num_steps is not None:
            return self.config.max_num_steps
        # 回退到 max_episode_steps
        if hasattr(self.config, 'max_episode_steps') and self.config.max_episode_steps is not None:
            return self.config.max_episode_steps
        # 默认值
        return 100

    def _dummy_observation(self) -> dict:
        frames = {
            key: np.zeros(self.config.image_shape, dtype=np.uint8)
            for key in self.config.camera_keys
        }
        return {"state": {"state": self._state.copy()}, "frames": frames}

    def _calc_step_reward(self, raw_reward: float, info: dict):
        """
        根据配置的奖励模式计算奖励值
        参考 maniskill_env.py 的 _calc_step_reward 实现
        """
        if self.reward_mode == "raw":
            # 直接返回原始奖励
            reward = torch.tensor([raw_reward], dtype=torch.float32)
        elif self.reward_mode == "only_success":
            # 只有成功才有奖励
            reward = torch.tensor([1.0 if self.success else 0.0], dtype=torch.float32)
        elif self.reward_mode == "dense":
            # 密集奖励，基于进度
            reward = torch.tensor([self.step_penalty], dtype=torch.float32)
            if self.success:
                reward += self.success_reward
            elif self.failure:
                reward += self.failure_penalty
        elif self.reward_mode == "dsrl_terminal":
            # 对齐 dsrl_pi0: 每步 -1，成功回合最后一步 0，失败回合最后一步仍为 -1。
            if self.success:
                reward = torch.tensor([self.dsrl_success_terminal_reward], dtype=torch.float32)
            elif self.failure:
                reward = torch.tensor([self.dsrl_failure_terminal_reward], dtype=torch.float32)
            else:
                reward = torch.tensor([self.dsrl_step_reward], dtype=torch.float32)
        else:  # 默认为稀疏奖励
            reward = torch.tensor([self.step_penalty], dtype=torch.float32)
            if self.success:
                reward = torch.tensor([self.success_reward], dtype=torch.float32)
            elif self.failure:
                reward = torch.tensor([self.failure_penalty], dtype=torch.float32)

        # 计算相对奖励
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _update_success_failure_status(self, action: np.ndarray, current_state: np.ndarray):
        """
        更新成功/失败状态的逻辑
        需要根据具体的任务定义实现
        """
        # 这里可以根据具体任务定义成功/失败条件
        # 示例：当末端执行器到达特定位置时认为成功
        # 这只是一个占位符，实际实现需要根据具体任务定制
        pass

    def _should_keyboard_evaluate(self) -> bool:
        """
        智能评估策略：基于训练阶段动态切换
        - 训练初期（global_step 小）：人工标注，积累高质量数据
        - 训练后期（global_step 大）：自动评估，加速流程
        
        配置方式：
        - auto_evaluate=True: 始终自动评估
        - auto_evaluate=False + keyboard_reward_wrapper: 始终人工评估
        - auto_evaluate='after_warmup': 预热后自动评估（需要设置 warmup_steps）
        """
        # === 模式 1: 显式启用自动评估 ===
        if hasattr(self.config, 'auto_evaluate') and self.config.auto_evaluate is True:
            return False
        
        # === 模式 2: 基于训练阶段智能切换 ===
        if hasattr(self.config, 'auto_evaluate') and self.config.auto_evaluate == 'after_warmup':
            # 检查是否过了预热期
            warmup_steps = getattr(self.config, 'warmup_steps', 10)
            
            # 尝试从 worker_info 获取 global_step
            if hasattr(self, 'worker_info') and self.worker_info is not None:
                if hasattr(self.worker_info, 'global_step'):
                    current_step = self.worker_info.global_step
                    if current_step >= warmup_steps:
                        return False  # 过了预热期，使用自动评估
            
            # 如果没有 worker_info，尝试从 config 获取
            if hasattr(self.config, 'global_step'):
                current_step = self.config.global_step
                if current_step >= warmup_steps:
                    return False
        
        # === 模式 3: 检查 keyboard_reward_wrapper ===
        if self.keyboard_reward_wrapper is not None:
            wrapper = str(self.keyboard_reward_wrapper).lower()
            if wrapper not in {"", "none", "null", "false"}:
                return True
        
        # === 模式 4: 检查 use_terminal_reward_wrapper ===
        if hasattr(self.config, 'use_terminal_reward_wrapper'):
            return bool(self.config.use_terminal_reward_wrapper)
        
        # 默认禁用人工评估
        return False

    def _query_episode_success(self) -> bool:
        """
        智能评估：支持自动模式（基于 return）和人工模式
        训练阶段使用自动评估，评估阶段可保留人工标注
        """
        # === 模式 1: 自动评估（基于 episode return）===
        if self.auto_evaluate:
            threshold = self.success_threshold
            # 使用累积的 episode_return
            episode_return = self.episode_return
            success = (episode_return > threshold)
            
            self.logger.info(
                f"[Auto-Eval] Return={episode_return:.2f}, "
                f"Threshold={threshold}, Success={success}"
            )
            return success
        
        # === 模式 2: 人工评估（默认）===
        if self._adapter is not None and hasattr(self._adapter, "evaluate_episode_result"):
            return bool(self._adapter.evaluate_episode_result())
        else:
            # 如果没有适配器（例如在 dummy 模式下），使用本地评估方法
            return bool(self.evaluate_episode_result())

    # 在 RosRobotEnv 类中添加评估方法
    def evaluate_episode_result(self):
        """
        在回合结束后通过键盘输入评估结果
        参考 train_utils_real.py 中的实现
        """
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            print("Episode finished. Mark as (1) Success or (0) Failure:")
            while True:
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    char_input = sys.stdin.read(1)
                    if char_input == '1':
                        print("Trial marked as SUCCESS.")
                        return True  # 成功
                    elif char_input == '0':
                        print("Trial marked as FAILURE.")                    
                        return False  # 失败
                    else:
                        print("Invalid input. Please enter '1' for Success or '0' for Failure:")
                time.sleep(0.01)  # 小延迟防止忙等待
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def reset(self, *, seed=None, options=None):
        self._num_steps = 0
        self.success = False
        self.failure = False
        self._episode_evaluated = False
        self.prev_step_reward = torch.zeros(1, dtype=torch.float32)
        self.episode_return = 0.0  # 累积 episode 的总 return
        
        if self.config.is_dummy:
            return self._dummy_observation(), {}
        obs, info = self._adapter.reset()
        
        # 添加成功/失败标识到 info
        info["success"] = self.success
        info["failure"] = self.failure
        info["is_success"] = self.success
        return obs, info

    def step(self, action: np.ndarray):
        self._num_steps += 1
        
        if not self.config.is_dummy:
            obs, raw_reward, terminated, truncated, info = self._adapter.step(action)
        else:
            obs, raw_reward, terminated, truncated, info = self._step_dummy(action)

        # [DEBUG] 在修改前打印从 _step_dummy 返回的原始值
        # print(f"[DEBUG] After _step_dummy call: truncated={truncated}, _num_steps={self._num_steps}")
        
        # 更新成功/失败状态
        self._update_success_failure_status(action, self._state)
        
        # 根据当前状态更新终止条件
        max_steps = self._get_max_steps()
        # print(f"[DEBUG] max_steps from _get_max_steps() = {max_steps}, checking: {self._num_steps} >= {max_steps} = {self._num_steps >= max_steps}")
        
        if self._num_steps >= max_steps:
            truncated = True
            # print(f"[DEBUG] Force setting truncated=True at step {self._num_steps}")

        is_terminal = bool(terminated or truncated)
        
        # # [DEBUG] 打印终止状态
        # print(f"[DEBUG] Step {self._num_steps}/{max_steps}: terminated={terminated}, truncated={truncated}, is_terminal={is_terminal}")
        # print(f"[DEBUG] _should_keyboard_evaluate() = {self._should_keyboard_evaluate()}")
        # print(f"[DEBUG] _episode_evaluated = {self._episode_evaluated}")
        
        # 在计算奖励之前先进行人工评估，确保最后一步的奖励正确
        if is_terminal and not self._episode_evaluated and self._should_keyboard_evaluate():
            # print("[DEBUG] Calling _query_episode_success()...")
            is_success = self._query_episode_success()
            self.success = bool(is_success)
            self.failure = not self.success
            self._episode_evaluated = True
            print(f"[INFO] Episode evaluated: success={self.success}, failure={self.failure}")
        
        # 计算奖励（使用更新后的 success/failure 标志）
        step_reward = self._calc_step_reward(raw_reward, info)
        
        # 累积 episode return（用于自动评估）
        self.episode_return += step_reward.item()
        
        # 更新 info 字典
        info["success"] = self.success
        info["failure"] = self.failure
        info["is_success"] = self.success
        info["episode_evaluated"] = self._episode_evaluated
        
        # 注意：episode 的总奖励应该在 environment worker 中累积计算
        # 这里只提供当前步的信息
        info["episode"] = {
            "episode_len": self._num_steps,
            "return": self.episode_return,  # 添加当前累积 return
        }

        return obs, step_reward.item(), terminated, truncated, info

    def _step_dummy(self, action: np.ndarray):
        """
        虚拟环境的 step 实现
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        copy_len = min(self.config.state_dim, self.config.action_dim)
        self._state[:copy_len] = action[:copy_len]
        if self.config.state_dim >= 7 and self.config.action_dim >= 6:
            pose = np.zeros((7,), dtype=np.float32)
            pose[:3] = action[:3] * self.config.action_scale[0]
            pose[3:] = R.from_euler("xyz", action[3:6] * self.config.action_scale[1]).as_quat()
            self._state[:7] = pose
        
        observation = self._dummy_observation()
        terminated = False
        max_steps = self._get_max_steps()
        truncated = self._num_steps >= max_steps
        
        # [DEBUG] 如果想快速测试评估，可以手动设置 terminated=True（仅用于测试）
        # terminated = (self._num_steps >= 5)  # ← 取消注释可在第 5 步强制终止测试
        
        return observation, 0.0, terminated, truncated, {"success": False, "failure": False}

    @property
    def task_description(self):
        return "Control a ROS robot with end-effector actions."