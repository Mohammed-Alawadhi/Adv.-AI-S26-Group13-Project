"""Pull wandb run histories to a local CSV for plotting."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from src.utils.config import RESULTS_DIR, WANDB_PROJECT


def fetch_run_history(entity_or_user, project=WANDB_PROJECT):
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{entity_or_user}/{project}")

    rows = []
    for run in runs:
        name = run.name
        if "seed99" in name or "size5" in name or "size4" in name:
            continue
        if name.startswith("sac_"):
            algo = "SAC"
        elif name.startswith("tdmpc2_"):
            algo = "TD-MPC2"
        else:
            continue
        try:
            seed = int(name.rsplit("seed", 1)[-1])
        except ValueError:
            continue

        print(f"  fetching {name}...")
        hist = run.history(samples=10_000, pandas=True)

        step_col = "global_step" if "global_step" in hist.columns else "_step"
        rew_cols = [c for c in ["rollout/ep_rew_mean", "rollout/ep_reward"]
                    if c in hist.columns]
        len_cols = [c for c in ["rollout/ep_len_mean", "rollout/ep_length"]
                    if c in hist.columns]

        for _, row in hist.iterrows():
            step = row.get(step_col)
            if pd.isna(step):
                continue
            rec = {
                "algo": algo, "seed": seed, "step": int(step),
                "ep_reward": next((row[c] for c in rew_cols if c in row and not pd.isna(row[c])), None),
                "ep_length": next((row[c] for c in len_cols if c in row and not pd.isna(row[c])), None),
                "collision": row.get("metrics/collision_rate", row.get("metrics/collision")),
                "success":   row.get("metrics/success_rate",   row.get("metrics/success")),
            }
            rows.append(rec)
    return pd.DataFrame(rows)


def extract_to_csv(entity_or_user, out_path=None):
    out_path = out_path or (RESULTS_DIR / "wandb_curves.csv")
    df = fetch_run_history(entity_or_user)
    df.to_csv(out_path, index=False)
    n = len(df)
    algos = df["algo"].unique().tolist() if not df.empty else []
    seeds = sorted(df["seed"].unique().tolist()) if not df.empty else []
    print(f"[ok] saved {n} rows -> {out_path}")
    print(f"     algos: {algos}")
    print(f"     seeds: {seeds}")
    return out_path
