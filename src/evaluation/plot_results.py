"""
Paper figures for the Option-2 narrative.

Fig 1: Transfer gap bar chart (HEADLINE — Section 5.1)
Fig 2: Per-environment success rate comparison
Fig 3: Per-seed roundabout success rate
Fig 4: Learning curves on highway-v0
Fig 5: Dynamics probe MSE per env (appendix)

All figures use a consistent color scheme:
  SAC      = #1f77b4 (blue)
  TD-MPC2  = #ff7f0e (orange)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path("/content/drive/MyDrive/tdmpc2-highway")
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLOR_SAC = "#1f77b4"
COLOR_TDM = "#ff7f0e"


def _ci95(arr):
    arr = np.asarray(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    t_crit = student_t.ppf(0.975, df=n - 1)
    return float(t_crit * np.std(arr, ddof=1) / np.sqrt(n))


def _load_results():
    eval_data = json.load(open(RESULTS_DIR / "eval_results.json"))
    return pd.DataFrame(eval_data)


# -----------------------------------------------------------------------------
# Figure 1: Transfer gap (HEADLINE)
# -----------------------------------------------------------------------------
def plot_transfer_gap(save: bool = True) -> Path:
    df = _load_results()

    # Compute per-seed retention: target_reward / training_reward
    retentions = {"SAC": {}, "TD-MPC2": {}}
    algos_map = {"SAC": "sac", "TD-MPC2": "tdmpc2"}
    for algo_label, algo_key in algos_map.items():
        train_rew = df[(df["algo"] == algo_key) & (df["env_id"] == "highway-v0")]["mean_reward"].values
        for env_id in ["merge-v0", "roundabout-v0"]:
            target_rew = df[(df["algo"] == algo_key) & (df["env_id"] == env_id)]["mean_reward"].values
            retention = target_rew / np.maximum(train_rew, 1e-6)
            retentions[algo_label][env_id] = retention

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    envs = ["merge-v0", "roundabout-v0"]
    x = np.arange(len(envs))
    width = 0.35

    sac_means = [100 * np.mean(retentions["SAC"][e]) for e in envs]
    sac_cis   = [100 * _ci95(retentions["SAC"][e])   for e in envs]
    tdm_means = [100 * np.mean(retentions["TD-MPC2"][e]) for e in envs]
    tdm_cis   = [100 * _ci95(retentions["TD-MPC2"][e])   for e in envs]

    ax.bar(x - width/2, sac_means, width, yerr=sac_cis, capsize=3,
           color=COLOR_SAC, label="SAC", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, tdm_means, width, yerr=tdm_cis, capsize=3,
           color=COLOR_TDM, label="TD-MPC2", edgecolor="black", linewidth=0.5)

    # Annotate exact values on top of bars
    for i, v in enumerate(sac_means):
        ax.text(i - width/2, v + 2, f"{v:.0f}%", ha="center", fontsize=7)
    for i, v in enumerate(tdm_means):
        ax.text(i + width/2, v + 2, f"{v:.0f}%", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel("Reward retention from highway-v0 (%)")
    ax.set_title("Cross-Action-Space Zero-Shot Transfer")
    ax.set_ylim(0, max(max(tdm_means + tdm_cis), 100) + 10)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(1.45, 102, "training-env baseline", fontsize=6, color="gray", ha="right")
    ax.legend(loc="upper left", frameon=True)

    if save:
        path = FIGURES_DIR / "fig1_transfer_gap.pdf"
        fig.savefig(path)
        fig.savefig(FIGURES_DIR / "fig1_transfer_gap.png")
        plt.close(fig)
        print(f"[ok] {path}")
        return path
    return fig


# -----------------------------------------------------------------------------
# Figure 2: Per-environment success rate
# -----------------------------------------------------------------------------
def plot_success_rates(save: bool = True) -> Path:
    df = _load_results()
    envs = ["highway-v0", "merge-v0", "roundabout-v0"]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    x = np.arange(len(envs))
    width = 0.35

    sac_means, sac_cis, tdm_means, tdm_cis = [], [], [], []
    for env_id in envs:
        s = df[(df["algo"] == "sac")    & (df["env_id"] == env_id)]["success_rate"].values
        t = df[(df["algo"] == "tdmpc2") & (df["env_id"] == env_id)]["success_rate"].values
        sac_means.append(100 * np.mean(s)); sac_cis.append(100 * _ci95(s))
        tdm_means.append(100 * np.mean(t)); tdm_cis.append(100 * _ci95(t))

    ax.bar(x - width/2, sac_means, width, yerr=sac_cis, capsize=3,
           color=COLOR_SAC, label="SAC", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, tdm_means, width, yerr=tdm_cis, capsize=3,
           color=COLOR_TDM, label="TD-MPC2", edgecolor="black", linewidth=0.5)

    for i, v in enumerate(sac_means):
        ax.text(i - width/2, v + 2, f"{v:.0f}%", ha="center", fontsize=7)
    for i, v in enumerate(tdm_means):
        ax.text(i + width/2, v + 2, f"{v:.0f}%", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success Rate by Environment")
    ax.set_ylim(0, max(max(sac_means + sac_cis), max(tdm_means + tdm_cis)) + 15)
    ax.legend(loc="upper right", frameon=True)

    # Vertical separator between training env and zero-shot envs
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.text(0.5, ax.get_ylim()[1] * 0.95, "  zero-shot →",
            fontsize=7, color="gray", ha="left", va="top")

    if save:
        path = FIGURES_DIR / "fig2_success_rates.pdf"
        fig.savefig(path); fig.savefig(FIGURES_DIR / "fig2_success_rates.png")
        plt.close(fig)
        print(f"[ok] {path}")
        return path
    return fig


# -----------------------------------------------------------------------------
# Figure 3: Per-seed roundabout success rate
# -----------------------------------------------------------------------------
def plot_per_seed_roundabout(save: bool = True) -> Path:
    df = _load_results()
    sub = df[df["env_id"] == "roundabout-v0"].sort_values(["algo", "seed"])

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    seeds = sorted(sub["seed"].unique())
    x = np.arange(len(seeds))
    width = 0.35

    sac_vals = sub[sub["algo"] == "sac"]["success_rate"].values * 100
    tdm_vals = sub[sub["algo"] == "tdmpc2"]["success_rate"].values * 100

    ax.bar(x - width/2, sac_vals, width, color=COLOR_SAC,
           label="SAC", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, tdm_vals, width, color=COLOR_TDM,
           label="TD-MPC2", edgecolor="black", linewidth=0.5)

    # Mean lines
    ax.axhline(np.mean(sac_vals), color=COLOR_SAC, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(np.mean(tdm_vals), color=COLOR_TDM, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(len(seeds) - 0.5, np.mean(sac_vals) - 2, f"SAC mean {np.mean(sac_vals):.1f}%",
            color=COLOR_SAC, fontsize=7, ha="right", va="top")
    ax.text(len(seeds) - 0.5, np.mean(tdm_vals) + 2, f"TD-MPC2 mean {np.mean(tdm_vals):.1f}%",
            color=COLOR_TDM, fontsize=7, ha="right", va="bottom")

    for i, v in enumerate(sac_vals):
        ax.text(i - width/2, v + 1, f"{v:.0f}%", ha="center", fontsize=7)
    for i, v in enumerate(tdm_vals):
        ax.text(i + width/2, v + 1, f"{v:.0f}%", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Roundabout-v0 Success Rate (per seed)")
    ax.set_ylim(0, max(max(sac_vals), max(tdm_vals)) + 15)
    ax.legend(loc="upper right", frameon=True)

    if save:
        path = FIGURES_DIR / "fig3_per_seed_roundabout.pdf"
        fig.savefig(path); fig.savefig(FIGURES_DIR / "fig3_per_seed_roundabout.png")
        plt.close(fig)
        print(f"[ok] {path}")
        return path
    return fig


# -----------------------------------------------------------------------------
# Figure 4: Learning curves on highway-v0
# -----------------------------------------------------------------------------
def plot_learning_curves(save: bool = True) -> Path:
    csv_path = RESULTS_DIR / "wandb_curves.csv"
    if not csv_path.exists():
        print(f"[skip] {csv_path} not found — Fig 4 cannot be rendered")
        return None

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ep_reward"])

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    bin_size = 5_000
    df["step_bin"] = (df["step"] // bin_size) * bin_size

    for algo, color in [("SAC", COLOR_SAC), ("TD-MPC2", COLOR_TDM)]:
        sub = df[df["algo"] == algo]
        if sub.empty:
            continue
        agg = sub.groupby("step_bin")["ep_reward"].agg(["mean", "std", "count"]).reset_index()
        agg = agg[agg["step_bin"] <= 100_000]
        x = agg["step_bin"].values
        y = agg["mean"].values
        n = agg["count"].values
        # 95% CI from std (multi-seed averages bin-wise)
        ci = 1.96 * agg["std"].fillna(0).values / np.sqrt(np.maximum(n, 1))

        ax.plot(x, y, color=color, label=algo, linewidth=1.5)
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.2)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.set_title("Learning Curves on highway-v0 (3 seeds, 95% CI)")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(0, 100_000)

    if save:
        path = FIGURES_DIR / "fig4_learning_curves.pdf"
        fig.savefig(path); fig.savefig(FIGURES_DIR / "fig4_learning_curves.png")
        plt.close(fig)
        print(f"[ok] {path}")
        return path
    return fig


# -----------------------------------------------------------------------------
# Figure 5: Dynamics probe (appendix)
# -----------------------------------------------------------------------------
def plot_dynamics_probe(save: bool = True) -> Path:
    probe_path = RESULTS_DIR / "dynamics_probe.json"
    if not probe_path.exists():
        print(f"[skip] {probe_path} not found — Fig 5 cannot be rendered")
        return None
    probe = json.load(open(probe_path))

    envs = ["highway-v0", "merge-v0", "roundabout-v0"]
    horizons = list(range(1, 6))

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    for env_id, marker in [("highway-v0", "o"), ("merge-v0", "s"), ("roundabout-v0", "^")]:
        sub = [p for p in probe if p["env_id"] == env_id]
        if not sub:
            continue
        # Average MSE per horizon across seeds
        mse_per_h = np.array([[p["model_mse_per_h"][h-1] for h in horizons] for p in sub])
        rand_per_h = np.array([[p["random_mse_per_h"][h-1] for h in horizons] for p in sub])
        ax.plot(horizons, mse_per_h.mean(0), marker=marker, label=f"model on {env_id}",
                linewidth=1.2, markersize=4)

    # Random baseline (averaged across all envs and seeds for visual clarity)
    all_random = np.array([p["random_mse_per_h"] for p in probe])
    ax.plot(horizons, all_random.mean(0), "k--", label="random baseline",
            linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Latent MSE (lower = better prediction)")
    ax.set_title("World Model Dynamics Probe (random-action trajectories)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=True, fontsize=7)

    if save:
        path = FIGURES_DIR / "fig5_dynamics_probe.pdf"
        fig.savefig(path); fig.savefig(FIGURES_DIR / "fig5_dynamics_probe.png")
        plt.close(fig)
        print(f"[ok] {path}")
        return path
    return fig


def render_all_figures():
    plot_transfer_gap()
    plot_success_rates()
    plot_per_seed_roundabout()
    plot_learning_curves()
    plot_dynamics_probe()
    print(f"\n[ok] All figures saved to {FIGURES_DIR}")
