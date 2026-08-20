"""Esegue DQN completo e le ablation target-network / experience-replay."""

import argparse
from dataclasses import replace

from breakout.config import DQNConfig
from breakout.training import ablation_config, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DQNConfig.total_steps)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu, cuda oppure scelta automatica")
    parser.add_argument("--only", choices=("full", "no_target", "no_replay", "neither"), default=None)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps deve essere positivo")

    base = replace(DQNConfig(), total_steps=args.steps)
    experiments = {
        "full": base,
        "no_target": ablation_config(base, replay=True, target=False),
        "no_replay": ablation_config(base, replay=False, target=True),
        "neither": ablation_config(base, replay=False, target=False),
    }
    for name, config in experiments.items():
        if args.only is None or args.only == name:
            run_training(name, config, seed=args.seed, device_name=args.device)


if __name__ == "__main__":
    main()
