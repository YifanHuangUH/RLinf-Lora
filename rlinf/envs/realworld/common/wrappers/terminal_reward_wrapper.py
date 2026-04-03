'''
Author: YifanHuangUH 1178010692@qq.com
Date: 2026-03-19 14:56:06
LastEditors: YifanHuangUH 1178010692@qq.com
LastEditTime: 2026-03-19 14:56:34
FilePath: /jack/hyf/lerobot/RLinf-yifan/rlinf/envs/realworld/common/wrappers/terminal_reward_wrapper.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import threading
import queue
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class TerminalInputListener:
    """Non-blocking terminal input listener."""
    
    def __init__(self):
        self.key_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        print("Terminal input listener started. Type a/b/c and press Enter.")
    
    def _listen(self):
        while self.running:
            try:
                user_input = input().strip().lower()
                if user_input:
                    self.key_queue.put(user_input)
            except EOFError:
                break
            except Exception:
                pass
    
    def get_key(self) -> str | None:
        if not self.key_queue.empty():
            return self.key_queue.get()
        return None
    
    def stop(self):
        self.running = False


class TerminalRewardDoneWrapper(gym.Wrapper):
    """Keyboard reward wrapper that works in pure terminal (no X11 required)."""
    
    def __init__(self, env: gym.Env, reward_mode: str = "always_replace"):
        super().__init__(env)
        self.reward_mode = reward_mode
        self.listener = TerminalInputListener()
        self._print_instructions()

    def _print_instructions(self):
        print("\n" + "="*60)
        print("TERMINAL REWARD CONTROL (No X11 required)")
        print("="*60)
        print("Type and press Enter:")
        print("  'a' - Failure (reward=-1, done=True)")
        print("  'b' - Continue (reward=0)")
        print("  'c' - Success (reward=+1, done=True)")
        print("="*60 + "\n")

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        last_intervened, updated_reward, updated_terminated = self._check_terminal()
        if last_intervened or self.reward_mode == "always_replace":
            reward = updated_reward
            if last_intervened:
                print(f">>> INTERVENTION: reward={reward}, done={updated_terminated} <<<")
        return observation, reward, updated_terminated, truncated, info

    def _check_terminal(self) -> tuple[bool, bool, float]:
        last_intervened = False
        done = False
        reward = 0.0
        
        key = self.listener.get_key()
        if key is None:
            return last_intervened, done, reward
        
        print(f"🎹 Terminal input received: '{key}'")
        
        if key == "a":
            reward = -1.0
            done = True
            last_intervened = True
            print("❌ FAILURE")
        elif key == "b":
            reward = 0.0
            last_intervened = True
            print("⏭️  CONTINUE")
        elif key == "c":
            reward = 1.0
            done = True
            last_intervened = True
            print("✅ SUCCESS")
        
        return last_intervened, done, reward

    def close(self):
        self.listener.stop()
        super().close()