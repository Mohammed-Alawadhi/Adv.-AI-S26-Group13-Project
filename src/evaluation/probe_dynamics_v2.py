"""Dynamics probe V2 — trained-policy trajectories."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.envs.highway_factory import make_highway_env, get_env_action_type
from src.utils.config import PROJECT_ROOT, RESULTS_DIR

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


def probe_dynamics_v2(seed=0, env_id="highway-v0", n_episodes=10, rollout_horizon=5):
    from tdmpc2 import TDMPC2
    from src.envs.tdmpc2_adapter import make_tdmpc2_highway_env
    from src.training.train_tdmpc2 import _build_cfg

    env_for_cfg = make_tdmpc2_highway_env(env_id="highway-v0", seed=seed)
    cfg = _build_cfg(env_for_cfg, seed=seed, total_timesteps=100_000, model_size=1)
    env_for_cfg.close()

    agent = TDMPC2(cfg)
    ckpt = PROJECT_ROOT / "checkpoints" / f"tdmpc2_highway-v0_seed{seed}" / "final.pt"
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    agent.model.load_state_dict(state["model"])
    agent.model.eval()
    device = next(agent.model.parameters()).device

    target_env = make_highway_env(env_id=env_id, seed=seed + 60_000)
    target_is_discrete = (get_env_action_type(env_id) == "discrete")

    errors_per_h = [[] for _ in range(rollout_horizon)]
    random_errors_per_h = [[] for _ in range(rollout_horizon)]

    for ep in range(n_episodes):
        obs, _ = target_env.reset(seed=seed + 60_000 + ep)
        traj_obs = [obs.astype(np.float32).copy()]
        traj_actions = []

        is_first = True
        done = False
        steps = 0
        while not done and steps < rollout_horizon + 10:
            obs_t = torch.from_numpy(obs.astype(np.float32))
            with torch.no_grad():
                cont_act = agent.act(obs_t, t0=is_first, eval_mode=True)
            cont_act_np = cont_act.detach().cpu().numpy().astype(np.float32)
            traj_actions.append(cont_act_np)

            step_act = _continuous_to_discrete(cont_act_np) if target_is_discrete else cont_act_np
            obs, _, term, trunc, _ = target_env.step(step_act)
            traj_obs.append(obs.astype(np.float32).copy())
            done = term or trunc
            is_first = False
            steps += 1

        if len(traj_actions) < rollout_horizon:
            continue

        with torch.no_grad():
            obs_tensor = torch.from_numpy(np.stack(traj_obs[:rollout_horizon + 1])).to(device)
            actions_tensor = torch.from_numpy(np.stack(traj_actions[:rollout_horizon])).to(device)
            actual_z = agent.model.encode(obs_tensor, task=None)
            z = actual_z[0:1]
            for h in range(rollout_horizon):
                a = actions_tensor[h:h+1]
                z_next_pred = agent.model.next(z, a, task=None)
                z_next_actual = actual_z[h+1:h+2]
                err = F.mse_loss(z_next_pred, z_next_actual).item()
                errors_per_h[h].append(err)
                rand_idx = torch.randperm(actual_z.shape[0])[0:1]
                rand_err = F.mse_loss(actual_z[rand_idx], z_next_actual).item()
                random_errors_per_h[h].append(rand_err)
                z = z_next_pred

    target_env.close()
    return {
        "seed": seed, "env_id": env_id,
        "n_episodes_used": min(n_episodes, len(errors_per_h[0])),
        "horizon": rollout_horizon,
        "model_mse_per_h":  [float(np.mean(e)) if e else 0.0 for e in errors_per_h],
        "random_mse_per_h": [float(np.mean(e)) if e else 0.0 for e in random_errors_per_h],
        "model_mse_std_per_h": [float(np.std(e)) if e else 0.0 for e in errors_per_h],
    }


def run_probe_v2_grid(seeds=(0, 1, 2), env_ids=("highway-v0", "merge-v0", "roundabout-v0")):
    results = []
    for env_id in env_ids:
        for seed in seeds:
            print(f"  V2 probing seed={seed}, env={env_id}...")
            r = probe_dynamics_v2(seed=seed, env_id=env_id, n_episodes=10, rollout_horizon=5)
            results.append(r)
            h1, h5 = r["model_mse_per_h"][0], r["model_mse_per_h"][-1]
            print(f"    model MSE @ h=1: {h1:.4f}, @ h=5: {h5:.4f}")
    out = RESULTS_DIR / "dynamics_probe_v2.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[ok] saved {len(results)} V2 probe results -> {out}")
    return results
