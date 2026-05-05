"""
Custom SB3 callback: log per-episode collision and success rates.
Pushed to wandb at every rollout end so paper plots can use them directly.
"""
from __future__ import annotations
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class SafetyMetricsCallback(BaseCallback):
    """Log collision rate, success rate, and episode length to wandb."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._collisions: list[bool] = []
        self._successes:  list[bool] = []
        self._lengths:    list[int]  = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_collision" in info:
                self._collisions.append(info["episode_collision"])
                self._successes.append(info["episode_success"])
                self._lengths.append(info["episode_length"])
        return True

    def _on_rollout_end(self) -> None:
        if not self._collisions:
            return
        metrics = {
            "metrics/collision_rate": float(np.mean(self._collisions)),
            "metrics/success_rate":   float(np.mean(self._successes)),
            "metrics/episode_length": float(np.mean(self._lengths)),
            "metrics/n_episodes":     len(self._collisions),
            "global_step":            self.num_timesteps,
        }
        if wandb.run is not None:
            wandb.log(metrics)
        if self.verbose:
            print(f"[step {self.num_timesteps}] {metrics}")
        self._collisions.clear()
        self._successes.clear()
        self._lengths.clear()
