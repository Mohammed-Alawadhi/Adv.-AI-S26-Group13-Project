"""
Zero-shot evaluation of trained SAC + TD-MPC2 agents.

Cross-action-space handling: continuous policy is binned to discrete
actions on merge/roundabout. t0 resets per-episode (only at env.reset
boundaries), preserving MPPI's warm-start.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.envs.highway_factory import make_highway_env, get_env_action_type
from src.utils.config import (
    CHECKPOINTS_DIR, EVAL_ENV_IDS, N_EVAL_EPISODES,
    PROJECT_ROOT, RESULTS_DIR, SEEDS,
)

TDMPC2_REPO = PROJECT_ROOT / "third_party" / "tdmpc2" / "tdmpc2"
if str(TDMPC2_REPO) not in sys.path:
    sys.path.insert(0, str(TDMPC2_REPO))


def _continuous_to_discrete(action):
    a = np.asarray(action).flatten()
    if a.size < 2: return 1
    steer, accel = float(a[0]), float(a[1])
    if steer < -0.4: return 0
    if steer > 0.4:  return 2
    if accel > 0.3:  return 3
    if accel < -0.3: return 4
    return 1


def load_sac(seed):
    from stable_baselines3 import SAC
    ckpt = CHECKPOINTS_DIR / f"sac_highway-v0_seed{seed}" / "final.zip"
    return SAC.load(str(ckpt), device="cuda" if torch.cuda.is_available() else "cpu")


def load_tdmpc2(seed, env_id="highway-v0"):
    from tdmpc2 import TDMPC2
    from src.envs.tdmpc2_adapter import make_tdmpc2_highway_env
    from src.training.train_tdmpc2 import _build_cfg

    env = make_tdmpc2_highway_env(env_id=env_id, seed=seed)
    cfg = _build_cfg(env, seed=seed, total_timesteps=100_000, model_size=1)
    agent = TDMPC2(cfg)
    ckpt_path = CHECKPOINTS_DIR / f"tdmpc2_highway-v0_seed{seed}" / "final.pt"
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    agent.model.load_state_dict(state["model"])
    agent.model.eval()
    env.close()
    return agent, cfg


def run_episodes(algo, seed, env_id, n_episodes=N_EVAL_EPISODES, deterministic=True):
    env = make_highway_env(env_id=env_id, seed=seed + 10_000)
    target_is_discrete = (get_env_action_type(env_id) == "discrete")

    if algo == "sac":
        agent = load_sac(seed)
        def policy(obs):
            action, _ = agent.predict(obs, deterministic=deterministic)
            return action
    elif algo == "tdmpc2":
        agent, _ = load_tdmpc2(seed, env_id="highway-v0")
        is_first_step = [True]
        def policy(obs):
            obs_t = torch.from_numpy(obs.astype(np.float32))
            with torch.no_grad():
                action = agent.act(obs_t, t0=is_first_step[0], eval_mode=True)
            is_first_step[0] = False
            return action.detach().cpu().numpy().astype(np.float32)
        policy._reset_t0 = lambda: is_first_step.__setitem__(0, True)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    rewards, lengths, collisions, successes = [], [], [], []
    start = time.time()
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + 10_000 + ep)
        if hasattr(policy, "_reset_t0"):
            policy._reset_t0()
        ep_reward = 0.0
        ep_length = 0
        ep_collided = False
        truncated = False
        terminated = False
        done = False
        while not done:
            action = policy(obs)
            if target_is_discrete:
                action = _continuous_to_discrete(action)
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            ep_length += 1
            if info.get("crashed", False):
                ep_collided = True
            done = terminated or truncated
        rewards.append(ep_reward)
        lengths.append(ep_length)
        collisions.append(ep_collided)
        successes.append((not ep_collided) and bool(truncated))

    env.close()
    elapsed = time.time() - start
    return {
        "algo": algo, "seed": seed, "env_id": env_id, "n_episodes": n_episodes,
        "rewards":    [float(x) for x in rewards],
        "lengths":    [int(x)   for x in lengths],
        "collisions": [bool(x)  for x in collisions],
        "successes":  [bool(x)  for x in successes],
        "mean_reward":    float(np.mean(rewards)),
        "std_reward":     float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "mean_length":    float(np.mean(lengths)),
        "collision_rate": float(np.mean(collisions)),
        "success_rate":   float(np.mean(successes)),
        "wall_time_sec":  float(elapsed),
    }


def run_full_evaluation(algos=("sac", "tdmpc2"), seeds=None, env_ids=None,
                        n_episodes=N_EVAL_EPISODES, out_path=None):
    seeds   = list(seeds   or SEEDS)
    env_ids = list(env_ids or EVAL_ENV_IDS)
    out_path = out_path or (RESULTS_DIR / "eval_results.json")

    n_combos = len(algos) * len(seeds) * len(env_ids)
    print(f"Eval grid: {len(algos)} algos x {len(seeds)} seeds x {len(env_ids)} envs "
          f"x {n_episodes} eps = {n_combos*n_episodes} episodes total")

    results = []
    combo_idx = 0
    for algo in algos:
        for seed in seeds:
            for env_id in env_ids:
                combo_idx += 1
                print(f"\n[{combo_idx}/{n_combos}] {algo} seed={seed} env={env_id}")
                r = run_episodes(algo, seed, env_id, n_episodes=n_episodes)
                results.append(r)
                rew, std = r["mean_reward"], r["std_reward"]
                crash    = 100 * r["collision_rate"]
                succ     = 100 * r["success_rate"]
                t        = r["wall_time_sec"]
                print(f"  -> reward={rew:6.2f}+/-{std:.2f}  "
                      f"crash={crash:5.1f}%  success={succ:5.1f}%  time={t:.1f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ok] saved {len(results)} entries -> {out_path}")
    return results
