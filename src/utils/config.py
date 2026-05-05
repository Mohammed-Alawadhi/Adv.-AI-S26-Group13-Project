"""Central configuration."""
from pathlib import Path

PROJECT_ROOT    = Path("/content/drive/MyDrive/tdmpc2-highway")
RESULTS_DIR     = PROJECT_ROOT / "results"
VIDEOS_DIR      = PROJECT_ROOT / "videos"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR        = PROJECT_ROOT / "logs"
for d in [RESULTS_DIR, VIDEOS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HIGHWAY_CONFIG = {
    "action": {"type": "ContinuousAction", "longitudinal": True, "lateral": True},
    "observation": {
        "type": "Kinematics", "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False, "normalize": True, "order": "sorted",
    },
    "simulation_frequency": 15, "policy_frequency": 5, "duration": 40,
    "vehicles_count": 20, "lanes_count": 4,
    "collision_reward": -1.0, "right_lane_reward": 0.1,
    "high_speed_reward": 0.4, "lane_change_reward": 0.0,
    "reward_speed_range": [20, 30], "normalize_reward": True,
    "offroad_terminal": True,
}

TRAIN_ENV_ID    = "highway-v0"
EVAL_ENV_IDS    = ["highway-v0", "merge-v0", "roundabout-v0"]
TOTAL_TIMESTEPS = 100_000
EVAL_INTERVAL   = 5_000
SEEDS           = [0, 1, 2]
N_EVAL_EPISODES = 30
WANDB_PROJECT   = "tdmpc2-highway"
WANDB_ENTITY    = None
