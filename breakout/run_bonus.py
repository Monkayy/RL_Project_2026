"""Esegue l'esperimento DQN con tutte le estensioni Bonus (Double, Dueling, PER)."""

import argparse
from dataclasses import replace

from breakout.config import DQNConfig
from breakout.training import run_training

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = replace(
        DQNConfig(),
        total_steps=args.steps,
        use_double_dqn=True,
        use_dueling=True,
        use_per=True
    )
    
    run_training("bonus_full", config, seed=args.seed, device_name=args.device)

if __name__ == "__main__":
    main()
