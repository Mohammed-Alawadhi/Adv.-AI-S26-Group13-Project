"""
Highway-env factory + wrappers.

Per-env action policy:
  highway-v0:    continuous (training)
  merge-v0:      discrete  (zero-shot)
  roundabout-v0: discrete  (zero-shot)
"""
from __future__ import annotations
import copy
from typing import Optional

import gymnasium as gym
import highway_env  # noqa: F401  registers env IDs
import numpy as np

from src.utils.config import HIGHWAY_CONFIG


_ENV_ACTION_TYPE = {
    "highway-v0":    "ContinuousAction",
    "merge-v0":      "DiscreteMetaAction",
    "roundabout-v0": "DiscreteMetaAction",
}


def _build_config(env_id, overrides):
    cfg = copy.deepcopy(HIGHWAY_CONFIG)
    action_type = _ENV_ACTION_TYPE.get(env_id, "ContinuousAction")
    if action_type == "ContinuousAction":
        cfg["action"] = {"type": "ContinuousAction", "longitudinal": True, "lateral": True}
    else:
        cfg["action"] = {"type": "DiscreteMetaAction"}
    if overrides:
        cfg.update(overrides)
    return cfg


class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low  = self.observation_space.low.flatten()
        high = self.observation_space.high.flatten()
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    def observation(self, obs):
        return obs.astype(np.float32).flatten()


class MetricsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._collided = False
        self._steps = 0
    def reset(self, **kwargs):
        self._collided = False
        self._steps = 0
        return self.env.reset(**kwargs)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        if info.get("crashed", False):
            self._collided = True
        if terminated or truncated:
            info["episode_collision"] = bool(self._collided)
            info["episode_success"]   = bool((not self._collided) and truncated)
            info["episode_length"]    = int(self._steps)
        return obs, reward, terminated, truncated, info


def make_highway_env(env_id="highway-v0", seed=None, render_mode=None, config_overrides=None):
    cfg = _build_config(env_id, config_overrides)
    env = gym.make(env_id, config=cfg, render_mode=render_mode)
    env = FlattenObservation(env)
    env = MetricsWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def get_env_action_type(env_id):
    return "continuous" if _ENV_ACTION_TYPE.get(env_id) == "ContinuousAction" else "discrete"
