"""
Wrapper for retraining TD-MPC2 with scaled-up hyperparameters.
Saves checkpoints to a *separate* directory (`tdmpc2_size5_seedN`)
so the original 1M results are preserved for comparison.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

TDMPC2_REPO = Path("/content/drive/MyDrive/tdmpc2-highway/third_party/tdmpc2/tdmpc2")
if str(TDMPC2_REPO) not in sys.path:
    sys.path.insert(0, str(TDMPC2_REPO))

from tdmpc2 import TDMPC2
from common.buffer import Buffer

from src.envs.tdmpc2_adapter import make_tdmpc2_highway_env
from src.training.train_tdmpc2 import _build_cfg, _to_td
from src.utils.config import (
    CHECKPOINTS_DIR, TRAIN_ENV_ID, WANDB_ENTITY, WANDB_PROJECT,
)
import time


def train_tdmpc2_scaled(
    seed: int,
    total_timesteps: int = 200_000,
    model_size: int = 5,
    horizon: int = 5,
    seed_steps_override: int = 2_000,
    use_wandb: bool = True,
    log_every: int = 4_000,
):
    """Retrain TD-MPC2 with scaled hyperparameters.
    Checkpoints go to checkpoints/tdmpc2_size{model_size}_seed{seed}/."""
    np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    run_name = f"tdmpc2_size{model_size}_h{horizon}_seed{seed}"
    ckpt_dir = CHECKPOINTS_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name,
            config={"algo": "TD-MPC2", "env_id": TRAIN_ENV_ID, "seed": seed,
                    "total_timesteps": total_timesteps, "model_size": model_size,
                    "horizon": horizon},
            reinit=True,
        )

    env = make_tdmpc2_highway_env(env_id=TRAIN_ENV_ID, seed=seed)
    cfg = _build_cfg(env, seed=seed, total_timesteps=total_timesteps,
                     model_size=model_size)
    # Set num_q default if MODEL_SIZE didn't provide one (size=5 only sets
    # enc_dim, mlp_dim, latent_dim, num_enc_layers — not num_q).
    if "num_q" not in cfg:
        cfg.num_q = 5  # paper default for size>=5
    # Override defaults
    cfg.horizon = horizon
    cfg.seed_steps = seed_steps_override
    cfg.buffer_size = min(int(total_timesteps), 200_000)

    agent = TDMPC2(cfg)
    buffer = Buffer(cfg)

    step = 0
    done = True
    tds: list = []
    info: dict = {}
    ep_rewards_recent, ep_lengths_recent, ep_collisions_recent = [], [], []
    t_start = time.time()

    while step <= total_timesteps:
        if done:
            if step > 0:
                ep_reward = float(torch.tensor([t["reward"] for t in tds[1:]]).sum())
                ep_length = len(tds)
                ep_collided = bool(info.get("episode_collision", False))
                ep_rewards_recent.append(ep_reward)
                ep_lengths_recent.append(ep_length)
                ep_collisions_recent.append(ep_collided)
                if use_wandb:
                    wandb.log({
                        "rollout/ep_reward": ep_reward,
                        "rollout/ep_length": ep_length,
                        "metrics/collision": int(ep_collided),
                        "metrics/success":   float(info.get("success", 0.0)),
                        "global_step":       step,
                    }, step=step)
                buffer.add(torch.cat(tds))
            obs = env.reset()
            tds = [_to_td(obs, env)]

        if step > cfg.seed_steps:
            action = agent.act(obs, t0=(len(tds) == 1))
        else:
            action = env.rand_act()

        obs, reward, done, info = env.step(action)
        tds.append(_to_td(obs, env, action, reward, info["terminated"]))

        if step >= cfg.seed_steps:
            n_updates = cfg.seed_steps if step == cfg.seed_steps else 1
            if step == cfg.seed_steps:
                print(f"[seed_steps={cfg.seed_steps} reached] pretraining...")
            for _ in range(n_updates):
                _ = agent.update(buffer)

        if (step + 1) % log_every == 0 and ep_rewards_recent:
            tail = slice(-20, None)
            elapsed = time.time() - t_start
            mean_rew = np.mean(ep_rewards_recent[tail])
            mean_len = np.mean(ep_lengths_recent[tail])
            crash_pct = 100 * np.mean(ep_collisions_recent[tail])
            sps = step / elapsed
            print(f"[step {step+1:6d}/{total_timesteps}] "
                  f"ep_rew(20)={mean_rew:6.2f} "
                  f"ep_len(20)={mean_len:5.1f} "
                  f"crash%(20)={crash_pct:3.0f} "
                  f"sps={sps:5.1f}")
        step += 1

    final_path = ckpt_dir / "final.pt"
    agent.save(str(final_path))
    print(f"\n[seed={seed} model_size={model_size}] saved -> {final_path}")
    if use_wandb and wandb.run is not None:
        wandb.finish()
    return final_path
