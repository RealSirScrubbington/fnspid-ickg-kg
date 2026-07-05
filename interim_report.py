"""Generate a go/no-go report from the trial's checkpoint files *while it is
still running* (reads partial stats/triplets; tolerant of a partial last line)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trial
from ickg_kg.config import Config, TRIPLET_DIR


def main(seed: int = 42, n: int = 500) -> None:
    cfg = Config(n=n, seed=seed)
    stats = TRIPLET_DIR / f"trial_stats_seed{seed}_n{n}.jsonl"
    csv_ = TRIPLET_DIR / f"trial_triplets_seed{seed}_n{n}.csv"
    if not stats.exists():
        print("no checkpoint yet:", stats)
        return
    trial.report(cfg, "trial_interim", stats, csv_, [])


if __name__ == "__main__":
    main()
