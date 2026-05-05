"""Adapter: highway-env -> TD-MPC2 TensorWrapper interface."""
from __future__ import annotations
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch

from src.envs.highway_factory import make_highway_env


class HighwayTDMPC2Wrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps=200):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps

    def rand_act(self):
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            if x.dtype == torch.float64:
                x = x.float()
        return x

    def reset(self, **kwargs):
        obs, _info = self.env.reset(**kwargs)
        return self._to_tensor(obs.astype(np.float32))

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy().astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        info = defaultdict(float, info)
        info["success"]    = float(info.get("episode_success", False))
        info["terminated"] = torch.tensor(float(terminated))
        return (
            self._to_tensor(obs.astype(np.float32)),
            torch.tensor(reward, dtype=torch.float32),
            done,
            info,
        )


def make_tdmpc2_highway_env(env_id="highway-v0", seed=0, max_episode_steps=200):
    base = make_highway_env(env_id=env_id, seed=seed)
    return HighwayTDMPC2Wrapper(base, max_episode_steps=max_episode_steps)
