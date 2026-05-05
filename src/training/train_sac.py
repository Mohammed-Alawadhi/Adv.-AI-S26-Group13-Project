"""
SAC baseline for highway-v0 (model-free).

train_sac(seed=N) trains one seed end-to-end. Loop over SEEDS in the
notebook driver to produce the multi-seed comparison.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from src.envs.highway_factory import make_highway_env
from src.training.callbacks import SafetyMetricsCallback
from src.utils.config import (
    CHECKPOINTS_DIR, EVAL_INTERVAL, TOTAL_TIMESTEPS,
    TRAIN_ENV_ID, WANDB_ENTITY, WANDB_PROJECT,
)


def _make_vec_env(env_id: str, seed: int, monitor_dir: Path) -> DummyVecEnv:
    """SB3 needs a VecEnv; we wrap a single env with Monitor for episode logging."""
    def _thunk():
        env = make_highway_env(env_id=env_id, seed=seed)
        return Monitor(env, filename=str(monitor_dir / f"sac_seed{seed}"))
    return DummyVecEnv([_thunk])


def train_sac(
    seed: int,
    total_timesteps: int = TOTAL_TIMESTEPS,
    env_id: str = TRAIN_ENV_ID,
    eval_interval: int = EVAL_INTERVAL,
    use_wandb: bool = True,
) -> Path:
    """Train one SAC seed; return path to the saved final model."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_name = f"sac_{env_id}_seed{seed}"
    ckpt_dir = CHECKPOINTS_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if use_wandb:
        wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY, name=run_name,
            config={"algo": "SAC", "env_id": env_id, "seed": seed,
                    "total_timesteps": total_timesteps},
            sync_tensorboard=True, reinit=True,
        )

    train_env = _make_vec_env(env_id, seed=seed,        monitor_dir=ckpt_dir)
    eval_env  = _make_vec_env(env_id, seed=seed + 1000, monitor_dir=ckpt_dir)

    # Hyperparameters: SB3 defaults work for highway-env. learning_starts
    # avoids premature updates from a near-empty buffer; gamma=0.95 reflects
    # episodes capped at 200 steps.
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=str(ckpt_dir / "tb"),
    )

    callbacks = [
        SafetyMetricsCallback(verbose=0),
        EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / "best"),
            log_path=str(ckpt_dir / "eval"),
            eval_freq=eval_interval,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        ),
    ]
    if use_wandb:
        callbacks.append(WandbCallback(
            gradient_save_freq=0,
            model_save_path=str(ckpt_dir / "wandb_models"),
            verbose=0,
        ))

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    final_path = ckpt_dir / "final.zip"
    model.save(str(final_path))
    print(f"[SAC seed={seed}] saved final → {final_path}")

    train_env.close(); eval_env.close()
    if use_wandb and wandb.run is not None:
        wandb.finish()

    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    train_sac(
        seed=args.seed,
        total_timesteps=args.steps,
        use_wandb=not args.no_wandb,
    )
