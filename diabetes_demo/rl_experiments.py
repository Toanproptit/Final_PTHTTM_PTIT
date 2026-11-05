"""Chạy trực tiếp thuật toán Python AGENT02; không dùng bản viết lại JavaScript."""
import hashlib
import json
from datetime import datetime, timezone
import matplotlib
matplotlib.use("Agg")
from legacy.agent02_experiment import (
    Config, run_baseline, run_learning_rate_experiment,
    run_exploration_experiment, run_multi_seed_experiment,
)
from .paths import RL, ROOT

def run_experiments(episodes: int = 5000, evaluation_episodes: int = 500) -> None:
    """Môi trường thương mại gốc là module giáo dục độc lập, không quyết định y khoa."""
    RL.mkdir(parents=True, exist_ok=True)
    config = Config(episodes=episodes, evaluation_episodes=evaluation_episodes, output_dir=str(RL))
    run_baseline(config, RL)
    run_learning_rate_experiment(config, RL)
    run_exploration_experiment(config, RL)
    run_multi_seed_experiment(config, RL)
    source = ROOT / "legacy" / "agent02_experiment.py"
    (RL / "manifest.json").write_text(json.dumps({
        "source": "legacy/agent02_experiment.py (copied byte-for-byte from agent02_code.tex)",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "original_sha256": hashlib.sha256((ROOT / "agent02_code.tex").read_bytes()).hexdigest(),
        "episodes": episodes, "steps_per_episode": config.steps_per_episode,
        "evaluation_episodes": evaluation_episodes, "seed": config.seed,
        "seeds": [11, 22, 33, 44, 55], "evaluation_exploration": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "independent e-commerce simulation, no clinical use",
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    run_experiments()
