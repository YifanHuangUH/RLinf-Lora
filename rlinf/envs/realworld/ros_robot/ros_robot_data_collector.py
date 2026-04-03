"""
ROS Robot Data Collector
用于收集真实机器人数据的包装器，模仿 dsrl_pi0 的数据收集流程
"""

import numpy as np
import sys
import select
import tty
import termios
import time
from typing import Dict, List, Tuple, Optional


class RosRobotDataCollector:
    """
    用于收集真实机器人数据的包装器，模仿 dsrl_pi0 的数据收集流程
    """
    def __init__(self, env, agent=None):
        self.env = env
        self.agent = agent
        
    def collect_trajectory(self, instruction="", max_steps=None):
        """
        收集一个轨迹，包括人工评估
        """
        obs, info = self.env.reset()
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'masks': [],  # 用于指示episode是否结束
            'next_observations': [],
            'is_success': False,
            'env_steps': 0,
        }
        
        step_count = 0
        while step_count < (max_steps or self.env.config.max_num_steps):
            # 获取智能体动作
            if self.agent:
                action = self.agent.act(obs)
            else:
                # 如果没有智能体，则可能需要手动控制或其他策略
                action = self.env.action_space.sample()
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 存储数据
            trajectory['observations'].append(obs)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['masks'].append(0.0 if terminated or truncated else 1.0)
            trajectory['next_observations'].append(next_obs)
            
            obs = next_obs
            step_count += 1
            
            if terminated or truncated:
                break
        
        # episode结束后进行人工评估
        is_success = self.evaluate_episode_result()
        
        trajectory['is_success'] = is_success
        trajectory['env_steps'] = step_count
        
        # 根据成功/失败更新奖励分布
        self._adjust_rewards_based_on_success(trajectory)
        
        return trajectory
    
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
    
    def _adjust_rewards_based_on_success(self, trajectory):
        """
        根据episode成功与否调整奖励分布
        参考 dsrl_pi0 中的奖励逻辑
        """
        if trajectory['is_success']:
            # 成功情况下，最后一步奖励设为0，其他为-1（类似dsrl_pi0）
            query_steps = len(trajectory['rewards'])
            rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            # 失败情况下，所有步骤都是-1
            query_steps = len(trajectory['rewards'])
            rewards = -np.ones(query_steps)
            masks = np.ones(query_steps)
        
        trajectory['rewards'] = rewards.tolist()
        trajectory['masks'] = masks.tolist()


def collect_multiple_trajectories(collector: RosRobotDataCollector, num_episodes: int, instruction: str = "") -> List[Dict]:
    """
    收集多个轨迹
    
    Args:
        collector: 数据收集器
        num_episodes: 要收集的episode数量
        instruction: 任务指令
    
    Returns:
        包含多个轨迹的列表
    """
    trajectories = []
    for episode_idx in range(num_episodes):
        print(f"Collecting episode {episode_idx + 1}/{num_episodes}")
        trajectory = collector.collect_trajectory(instruction)
        trajectories.append(trajectory)
        print(f"Episode {episode_idx + 1} collected. Success: {trajectory['is_success']}")
    
    return trajectories


def add_trajectory_to_buffer(trajectory: Dict, buffer) -> None:
    """
    将轨迹数据添加到经验回放缓冲区
    
    Args:
        trajectory: 轨迹数据
        buffer: 经验回放缓冲区
    """
    # 将轨迹数据添加到缓冲区
    for i in range(len(trajectory['observations'])):
        obs = trajectory['observations'][i]
        action = trajectory['actions'][i]
        reward = trajectory['rewards'][i]
        mask = trajectory['masks'][i]
        next_obs = trajectory['next_observations'][i]
        
        # 插入到缓冲区
        buffer.insert({
            'observations': obs,
            'actions': action,
            'rewards': reward,
            'masks': mask,
            'next_observations': next_obs
        })