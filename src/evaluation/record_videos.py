"""Demo video recorder with overlay annotations."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from src.envs.highway_factory import make_highway_env, get_env_action_type
from src.utils.config import CHECKPOINTS_DIR, PROJECT_ROOT, VIDEOS_DIR, SEEDS

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


def _annotate_frame(frame, text_lines, highlight_collision=False):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = font_s = ImageFont.load_default()

    draw.rectangle([(0, 0), (img.width, 30)], fill="white")
    draw.text((10, 5), text_lines[0], fill="black", font=font)

    bottom_h = 50
    draw.rectangle([(0, img.height - bottom_h), (img.width, img.height)], fill="white")
    for i, line in enumerate(text_lines[1:]):
        draw.text((10, img.height - bottom_h + 5 + i * 16),
                  line, fill="black", font=font_s)

    if highlight_collision:
        draw.rectangle([(0, 0), (img.width - 1, img.height - 1)],
                       outline="red", width=6)
    return np.array(img)


def _load_sac(seed):
    from stable_baselines3 import SAC
    ckpt = CHECKPOINTS_DIR / f"sac_highway-v0_seed{seed}" / "final.zip"
    return SAC.load(str(ckpt),
                    device="cuda" if torch.cuda.is_available() else "cpu")


def _load_tdmpc2(seed):
    from tdmpc2 import TDMPC2
    from src.envs.tdmpc2_adapter import make_tdmpc2_highway_env
    from src.training.train_tdmpc2 import _build_cfg

    env = make_tdmpc2_highway_env(env_id="highway-v0", seed=seed)
    cfg = _build_cfg(env, seed=seed, total_timesteps=100_000, model_size=1)
    agent = TDMPC2(cfg)
    state = torch.load(
        str(CHECKPOINTS_DIR / f"tdmpc2_highway-v0_seed{seed}" / "final.pt"),
        map_location="cpu", weights_only=False,
    )
    agent.model.load_state_dict(state["model"])
    agent.model.eval()
    env.close()
    return agent


def record_episode(algo, env_id, seed=0, max_steps=200, episode_seed_offset=42):
    env = make_highway_env(env_id=env_id, seed=seed + episode_seed_offset,
                           render_mode="rgb_array")
    target_is_discrete = (get_env_action_type(env_id) == "discrete")

    if algo == "sac":
        agent = _load_sac(seed)
        def policy(obs, first):
            action, _ = agent.predict(obs, deterministic=True)
            return action
    else:
        agent = _load_tdmpc2(seed)
        def policy(obs, first):
            obs_t = torch.from_numpy(obs.astype(np.float32))
            with torch.no_grad():
                a = agent.act(obs_t, t0=first, eval_mode=True)
            return a.detach().cpu().numpy().astype(np.float32)

    obs, info = env.reset(seed=seed + episode_seed_offset)
    frames = []
    total_reward = 0.0
    collided = False
    step = 0
    is_first = True

    while step < max_steps:
        action = policy(obs, is_first)
        is_first = False
        env_action = _continuous_to_discrete(action) if target_is_discrete else action

        frame = env.render()
        title = algo.upper() + "  on  " + env_id
        line2 = "Step: " + str(step).rjust(3) + "    Reward: " + format(total_reward, "6.2f")
        line3 = "COLLISION" if collided else "OK"
        annotated = _annotate_frame(frame, [title, line2, line3],
                                    highlight_collision=collided)
        frames.append(annotated)

        obs, r, term, trunc, info = env.step(env_action)
        total_reward += r
        step += 1
        if info.get("crashed", False):
            collided = True
        if term or trunc:
            frame = env.render()
            line3 = "COLLISION (END)" if collided else ("SUCCESS" if trunc else "END")
            line2 = "Step: " + str(step).rjust(3) + "    Reward: " + format(total_reward, "6.2f")
            frames.append(_annotate_frame(frame, [title, line2, line3],
                                          highlight_collision=collided))
            break

    env.close()
    return frames, float(total_reward), collided, step


def record_demo_videos(seeds=(0,), algos=("sac", "tdmpc2"),
                       env_ids=("highway-v0", "merge-v0", "roundabout-v0"),
                       fps=15, n_attempts=3):
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for algo in algos:
        for env_id in env_ids:
            print()
            print("=== Recording " + algo.upper() + " on " + env_id + " ===")
            best = None
            for attempt in range(n_attempts):
                seed = seeds[0]
                frames, total_r, collided, length = record_episode(
                    algo, env_id, seed=seed,
                    episode_seed_offset=42 + attempt * 7,
                )
                score = (0 if collided else 1, total_r, length)
                print("  attempt " + str(attempt+1) + ": "
                      "len=" + str(length) + "  "
                      "reward=" + format(total_r, "6.2f") + "  "
                      "crashed=" + str(collided))
                if best is None or score > best["score"]:
                    best = {"frames": frames, "reward": total_r,
                            "collided": collided, "length": length,
                            "score": score}

            out = VIDEOS_DIR / (algo + "_" + env_id + ".mp4")
            imageio.mimsave(str(out), best["frames"], fps=fps,
                            macro_block_size=1, quality=8)
            print("  saved " + str(out) + "  (" + str(len(best["frames"])) + " frames)")
            saved.append({
                "algo": algo, "env_id": env_id,
                "reward": best["reward"], "collided": best["collided"],
                "length": best["length"], "path": str(out),
            })
    return saved


def build_highlight_reel(saved_videos, fps=15, gap_seconds=0.5):
    out_path = VIDEOS_DIR / "highlight_reel.mp4"

    def title_card(text_lines, w=600, h=400):
        img = Image.new("RGB", (w, h), "black")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except Exception:
            font = ImageFont.load_default()
        y = h // 2 - len(text_lines) * 20
        for line in text_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((w - tw) // 2, y), line, fill="white", font=font)
            y += 40
        return np.array(img)

    all_frames = []
    intro = title_card(["TD-MPC2 vs SAC", "Highway-Env Demo"])
    all_frames.extend([intro] * int(2 * fps))

    for v in saved_videos:
        crashed_str = "CRASHED" if v["collided"] else "OK"
        card_lines = [
            v["algo"].upper(),
            v["env_id"],
            "reward " + format(v["reward"], ".1f") + "  |  " + crashed_str,
        ]
        clip = imageio.mimread(v["path"])
        if clip:
            h, w = clip[0].shape[:2]
            card = title_card(card_lines, w=w, h=h)
            all_frames.extend([card] * int(1.5 * fps))
            all_frames.extend(clip)
            all_frames.extend([clip[-1]] * int(gap_seconds * fps))

    imageio.mimsave(str(out_path), all_frames, fps=fps,
                    macro_block_size=1, quality=8)
    print()
    print("[ok] highlight reel: " + str(out_path)
          + "  (" + str(len(all_frames)) + " frames, "
          + format(len(all_frames)/fps, ".1f") + "s)")
    return out_path
