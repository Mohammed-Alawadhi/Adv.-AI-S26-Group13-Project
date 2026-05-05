"""TD-MPC2 — only _build_cfg and _to_td used here at eval time."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

TDMPC2_REPO = Path("/content/drive/MyDrive/tdmpc2-highway/third_party/tdmpc2/tdmpc2")
if str(TDMPC2_REPO) not in sys.path:
    sys.path.insert(0, str(TDMPC2_REPO))

from common import MODEL_SIZE


def _build_cfg(env, seed, total_timesteps, model_size=1):
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    ep_len  = env.max_episode_steps

    cfg = {
        "task": "highway-v0", "obs": "state",
        "obs_shape": {"state": (obs_dim,)},
        "action_dim": act_dim,
        "episode_length": ep_len,
        "episode_lengths": [ep_len],
        "seed_steps": max(1000, 5 * ep_len),
        "episodic": True,
        "multitask": False, "tasks": ["highway-v0"], "task_dim": 0,
        "steps": total_timesteps, "seed": seed,
        "batch_size": 256, "lr": 3e-4, "enc_lr_scale": 0.3,
        "grad_clip_norm": 20.0, "tau": 0.01, "rho": 0.5,
        "consistency_coef": 20.0, "reward_coef": 0.1, "value_coef": 0.1,
        "termination_coef": 1.0,
        "discount_denom": 5, "discount_min": 0.95, "discount_max": 0.995,
        "buffer_size": min(100_000, total_timesteps),
        "mpc": True, "horizon": 3, "iterations": 6,
        "num_samples": 512, "num_elites": 64, "num_pi_trajs": 24,
        "min_std": 0.05, "max_std": 2.0, "temperature": 0.5,
        "log_std_min": -10, "log_std_max": 2, "entropy_coef": 1e-4,
        "num_bins": 101, "vmin": -10.0, "vmax": 10.0,
        "model_size": model_size, "num_channels": 32,
        "dropout": 0.01, "simnorm_dim": 8,
        "compile": False, "save_video": False, "save_agent": False,
        "enable_wandb": False, "wandb_silent": True, "save_csv": False,
    }
    cfg["bin_size"] = (cfg["vmax"] - cfg["vmin"]) / (cfg["num_bins"] - 1)
    for k, v in MODEL_SIZE[model_size].items():
        cfg[k] = v
    if "num_q" not in cfg:
        cfg["num_q"] = 5
    return OmegaConf.create(cfg)


def _to_td(obs, env, action=None, reward=None, terminated=None):
    if isinstance(obs, dict):
        obs = TensorDict(obs, batch_size=(), device="cpu")
    else:
        obs = obs.unsqueeze(0).cpu()
    if action is None:
        action = torch.full_like(env.rand_act(), float("nan"))
    if reward is None:
        reward = torch.tensor(float("nan"))
    if terminated is None:
        terminated = torch.tensor(float("nan"))
    return TensorDict(
        obs=obs, action=action.unsqueeze(0),
        reward=reward.unsqueeze(0),
        terminated=terminated.unsqueeze(0),
        batch_size=(1,),
    )
