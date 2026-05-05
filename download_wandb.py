"""
Download EVERYTHING from the wandb project for the GitHub repo.

Pulls every run's metadata, summary, config, and full per-step history.
Saves into results/wandb/ in a structure that's easy to commit and review.

Usage:
    cd /Users/error/Downloads/Adv.-AI-S26-Group13-Project
    pip install wandb pandas
    python download_wandb.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any

import wandb
import pandas as pd

# --- Config ---
ENTITY  = "error1rak-american-university-of-sharjah"
PROJECT = "tdmpc2-highway"
ROOT    = Path(__file__).parent.resolve()
OUT_DIR = ROOT / "results" / "wandb"


def _safe(name: str) -> str:
    """Make a string safe to use as a folder name."""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _stringify(obj: Any) -> Any:
    """Make a value JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _stringify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    try:
        return str(obj)
    except Exception:
        return None


def main() -> None:
    print(f"Output directory: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Login (uses cached credentials from previous wandb login)
    wandb.login()

    # Get all runs in the project
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    print(f"\nFound {len(runs)} runs in {ENTITY}/{PROJECT}")

    # Index of all runs (one summary file)
    index = []

    for i, run in enumerate(runs, 1):
        safe_name = _safe(run.name)
        run_dir = OUT_DIR / f"{i:02d}_{safe_name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}/{len(runs)}] {run.name}  (state={run.state})")

        # 1. Metadata + config + summary as JSON
        meta = {
            "run_id":       run.id,
            "name":         run.name,
            "state":        run.state,
            "created_at":   str(run.created_at),
            "url":          run.url,
            "tags":         list(run.tags) if run.tags else [],
            "config":       _stringify(dict(run.config)),
            "summary":      _stringify(dict(run.summary)),
            "system_metrics": _stringify(dict(run.systemMetrics)) if hasattr(run, "systemMetrics") else {},
        }
        meta_path = run_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"    saved metadata.json")

        # 2. Full per-step history → CSV
        try:
            hist = run.history(samples=50_000, pandas=True)
            if not hist.empty:
                hist_path = run_dir / "history.csv"
                hist.to_csv(hist_path, index=False)
                print(f"    saved history.csv  ({len(hist)} rows, {len(hist.columns)} cols)")
            else:
                print(f"    history is empty, skipped")
        except Exception as e:
            print(f"    WARNING: failed to download history: {e}")

        # 3. (Optional) the wandb-metadata.json file from the run's filesystem
        try:
            for f in run.files():
                if f.name == "wandb-metadata.json":
                    f.download(root=str(run_dir), exist_ok=True, replace=True)
                    print(f"    saved wandb-metadata.json (from run files)")
                    break
        except Exception as e:
            pass  # not critical

        # Add to top-level index
        index.append({
            "i":            i,
            "name":         run.name,
            "state":        run.state,
            "run_id":       run.id,
            "created_at":   str(run.created_at),
            "url":          run.url,
            "folder":       run_dir.name,
            "final_step":   run.summary.get("global_step", run.summary.get("_step", None)),
            "final_reward": run.summary.get("rollout/ep_reward", None),
            "final_length": run.summary.get("rollout/ep_length", None),
        })

    # Save index
    index_path = OUT_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, default=str)
    print(f"\n[ok] Saved top-level index: {index_path}")

    # Also save as readable markdown
    md_path = OUT_DIR / "RUNS.md"
    with open(md_path, "w") as f:
        f.write("# W&B Runs Index\n\n")
        f.write(f"All runs from [`{ENTITY}/{PROJECT}`](https://wandb.ai/{ENTITY}/{PROJECT}).\n\n")
        f.write("| # | Run name | State | Final step | Final reward | Folder |\n")
        f.write("|---|---|---|---:|---:|---|\n")
        for r in index:
            step = f"{int(r['final_step']):,}" if isinstance(r['final_step'], (int, float)) else "—"
            rew  = f"{r['final_reward']:.2f}" if isinstance(r['final_reward'], (int, float)) else "—"
            f.write(f"| {r['i']:02d} | `{r['name']}` | {r['state']} | {step} | {rew} | [`{r['folder']}/`]({r['folder']}/) |\n")
    print(f"[ok] Saved readable index:  {md_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Done. Downloaded {len(runs)} runs to {OUT_DIR}/")
    print(f"{'='*60}")
    total_size = 0
    for p in OUT_DIR.rglob("*"):
        if p.is_file():
            total_size += p.stat().st_size
    print(f"  Total size: {total_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
