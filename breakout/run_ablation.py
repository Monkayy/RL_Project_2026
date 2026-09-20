"""Esegue gli esperimenti DQN: ablation (full, no_target, no_replay, neither) e bonus (Double, Dueling, PER)."""

import argparse
from dataclasses import replace

from breakout.config import DQNConfig
from breakout.training import ablation_config, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DQNConfig.total_steps)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu, cuda oppure scelta automatica")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=("full", "no_target", "no_replay", "neither", "bonus_full", "ddqn", "dueling", "per"),
        default=None,
        help="esegue un solo esperimento; se omesso esegue tutto (ablation + bonus)",
    )
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps deve essere positivo")

    base = replace(DQNConfig(), total_steps=args.steps)
    experiments = {
        "full": base,
        "no_target": ablation_config(base, replay=True, target=False),
        "no_replay": ablation_config(base, replay=False, target=True),
        "neither": ablation_config(base, replay=False, target=False),
        "ddqn": replace(base, use_double_dqn=True),
        "dueling": replace(base, use_dueling=True),
        "per": replace(base, use_per=True),
        "bonus_full": replace(base, use_double_dqn=True, use_dueling=True, use_per=True),
    }
    for name, config in experiments.items():
        if args.only is None or name in args.only:
            run_training(name, config, seed=args.seed, device_name=args.device)


if __name__ == "__main__":
    main()