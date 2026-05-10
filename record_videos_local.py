#!/usr/bin/env python3
"""
Local video recorder — runs on a developer machine in the project's venv.

Records 6 demo videos (2 algorithms × 3 environments) and saves them to ./videos/.
Each video shows one trained agent on one environment, with overlay text
showing algorithm name, current step, running reward, and a red border on
collision frames.

Usage
-----
    cd <repo root>
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    git clone https://github.com/nicklashansen/tdmpc2.git third_party/tdmpc2
    # On Apple Silicon / CPU-only machines, also patch TD-MPC2's hardcoded CUDA
    # references — see the patch script in README.md ("Local reproduction").
    python record_videos_local.py

Notes
-----
- Auto-detects CUDA / MPS / CPU. SAC runs on whichever is available.
- TD-MPC2 is forced to CPU because its codebase has hardcoded device choices
  and not all of its ops have MPS implementations in PyTorch.
- Skip-if-exists: re-running won't redo videos already in ./videos/.
"""
from __future__ import annotations
import gc
import os
import sys
from pathlib import Path

# Hide CUDA from torch entirely so TD-MPC2's hardcoded `cuda:0` falls back to
# CPU via our patches (see README "Local reproduction"). This also keeps SAC
# off CUDA on dev boxes where Torch *does* see CUDA but we want CPU/MPS only.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent.resolve()
VIDEOS_DIR = ROOT / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
TDMPC2_REPO = ROOT / "third_party" / "tdmpc2" / "tdmpc2"

if not TDMPC2_REPO.exists():
    print(f"ERROR: TD-MPC2 codebase not found at {TDMPC2_REPO}")
    print("       Clone it with:")
    print("       git clone https://github.com/nicklashansen/tdmpc2.git third_party/tdmpc2")
    sys.exit(1)

for p in [str(ROOT), str(TDMPC2_REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

assert np.__version__.startswith("1."), \
    f"NumPy is {np.__version__}; need 1.x. Run: pip install 'numpy==1.26.4'"

print(f"Project root: {ROOT}")
print(f"Videos will be saved to: {VIDEOS_DIR}")
device = ("cuda" if torch.cuda.is_available()
          else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Device (for SAC): {device}")
print()


from src.envs.highway_factory import make_highway_env, get_env_action_type


def continuous_to_discrete(action):
    """Map continuous (steer, accel) to highway-env's 5-action discrete space."""
    a = np.asarray(action).flatten()
    if a.size < 2:
        return 1
    s, ac = float(a[0]), float(a[1])
    if s < -0.4: return 0   # LANE_LEFT
    if s > 0.4:  return 2   # LANE_RIGHT
    if ac > 0.3: return 3   # FASTER
    if ac < -0.3: return 4  # SLOWER
    return 1                # IDLE


def annotate(frame, lines, collided=False):
    """Add white title/info bars and a red collision border."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
        font_s = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = font_s = ImageFont.load_default()
    draw.rectangle([(0, 0), (img.width, 30)], fill="white")
    draw.text((10, 5), lines[0], fill="black", font=font)
    bot = 50
    draw.rectangle([(0, img.height - bot), (img.width, img.height)], fill="white")
    for i, line in enumerate(lines[1:]):
        draw.text((10, img.height - bot + 5 + i * 16), line, fill="black", font=font_s)
    if collided:
        draw.rectangle([(0, 0), (img.width - 1, img.height - 1)],
                       outline="red", width=6)
    return np.array(img)


def record_one(policy_fn, label, env_id, seed_offset, max_steps=200):
    """Run one episode with a pre-loaded policy."""
    env = make_highway_env(env_id=env_id, seed=seed_offset, render_mode="rgb_array")
    target_is_discrete = (get_env_action_type(env_id) == "discrete")
    obs, _ = env.reset(seed=seed_offset)
    frames, total_r, collided, step = [], 0.0, False, 0
    is_first = True
    while step < max_steps:
        action = policy_fn(obs, is_first)
        is_first = False
        env_a = continuous_to_discrete(action) if target_is_discrete else action
        frame = env.render()
        title = label + "  on  " + env_id
        line2 = f"Step: {step:3d}    Reward: {total_r:6.2f}"
        line3 = "COLLISION" if collided else "OK"
        frames.append(annotate(frame, [title, line2, line3], collided=collided))
        obs, r, term, trunc, info = env.step(env_a)
        total_r += r
        step += 1
        if info.get("crashed", False):
            collided = True
        if term or trunc:
            frame = env.render()
            line3 = "COLLISION (END)" if collided else ("SUCCESS" if trunc else "END")
            line2 = f"Step: {step:3d}    Reward: {total_r:6.2f}"
            frames.append(annotate(frame, [title, line2, line3], collided=collided))
            break
    env.close()
    return frames, float(total_r), collided, step


# Load both agents ONCE
print("Loading SAC seed=0...", flush=True)
from stable_baselines3 import SAC
sac_agent = SAC.load(str(ROOT / "checkpoints/sac_highway-v0_seed0/final.zip"),
                     device=device)
def sac_policy(obs, _is_first):
    a, _ = sac_agent.predict(obs, deterministic=True)
    return a
print("  ✓ SAC loaded", flush=True)

print("Loading TD-MPC2 seed=0 (forcing CPU)...", flush=True)
from tdmpc2 import TDMPC2
from src.envs.tdmpc2_adapter import make_tdmpc2_highway_env
from src.training.train_tdmpc2 import _build_cfg
from omegaconf import OmegaConf

tmp_env = make_tdmpc2_highway_env(env_id="highway-v0", seed=0)
cfg = _build_cfg(tmp_env, seed=0, total_timesteps=100_000, model_size=1)
OmegaConf.set_struct(cfg, False)
cfg.device = "cpu"
tdm_agent = TDMPC2(cfg)
state = torch.load(str(ROOT / "checkpoints/tdmpc2_highway-v0_seed0/final.pt"),
                   map_location="cpu", weights_only=False)
tdm_agent.model.load_state_dict(state["model"])
tdm_agent.model.eval()
tmp_env.close()
def tdm_policy(obs, is_first):
    obs_t = torch.from_numpy(obs.astype(np.float32))
    with torch.no_grad():
        a = tdm_agent.act(obs_t, t0=is_first, eval_mode=True)
    return a.detach().cpu().numpy().astype(np.float32)
print("  ✓ TD-MPC2 loaded", flush=True)
print()


# Record all 6 combos
saved = []
for algo, label, policy in [("sac", "SAC", sac_policy),
                            ("tdmpc2", "TD-MPC2", tdm_policy)]:
    for env_id in ["highway-v0", "merge-v0", "roundabout-v0"]:
        out = VIDEOS_DIR / f"{algo}_{env_id}.mp4"
        if out.exists() and out.stat().st_size > 50_000:
            print(f"[skip] {out.name} already exists ({out.stat().st_size/1e6:.1f} MB)",
                  flush=True)
            saved.append({"algo": algo, "env_id": env_id, "path": str(out),
                          "reward": 0.0, "collided": False, "length": 0})
            continue

        print(f"=== {label} on {env_id} ===", flush=True)
        best = None
        for attempt in range(3):
            frames, r, collided, length = record_one(
                policy, label, env_id, seed_offset=42 + attempt * 7,
            )
            score = (0 if collided else 1, r, length)
            print(f"  attempt {attempt+1}: len={length:3d}  reward={r:6.2f}  crashed={collided}",
                  flush=True)
            if best is None or score > best["score"]:
                if best is not None:
                    del best["frames"]
                best = {"frames": frames, "reward": r, "collided": collided,
                        "length": length, "score": score}
            else:
                del frames
            gc.collect()

        imageio.mimsave(str(out), best["frames"], fps=15,
                        macro_block_size=1, quality=8)
        print(f"  ✓ saved {out.name}  ({len(best['frames'])} frames, "
              f"{out.stat().st_size/1e6:.1f} MB)", flush=True)
        saved.append({
            "algo": algo, "env_id": env_id,
            "reward": best["reward"], "collided": best["collided"],
            "length": best["length"], "path": str(out),
        })
        del best
        gc.collect()

print()
print("=" * 60, flush=True)
print(f"  Done — {len(saved)} videos in {VIDEOS_DIR}/", flush=True)
print("=" * 60, flush=True)
for v in saved:
    flag = "❌" if v["collided"] else "✓ "
    print(f"  {flag} {v['algo']:8s} {v['env_id']:18s}  reward={v['reward']:6.2f}  len={v['length']:3d}",
          flush=True)
