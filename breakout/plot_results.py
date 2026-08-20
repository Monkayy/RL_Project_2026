"""Grafici comparativi per gli esperimenti DQN su Breakout."""

import argparse
import matplotlib.pyplot as plt
import numpy as np

from breakout.config import DATA_DIR, FIGURE_DIR


def moving_mean(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) == 0:
        return values
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def load_results(seed: int):
    results = {}
    for name in ("full", "no_target", "no_replay", "neither"):
        path = DATA_DIR / f"{name}_seed_{seed}.npz"
        if path.exists():
            results[name] = np.load(path, allow_pickle=True)
    if not results:
        raise FileNotFoundError("Nessun risultato trovato: esegui prima breakout.run_ablation.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = load_results(args.seed)
    labels = {"full": "DQN completo", "no_target": "senza target network",
              "no_replay": "senza experience replay", "neither": "senza entrambe"}

    plt.figure(figsize=(9, 5))
    for name, result in results.items():
        curve = moving_mean(result["episode_returns"])
        if len(curve):
            plt.plot(np.arange(len(curve)), curve, label=labels[name])
    plt.xlabel("Episodio")
    plt.ylabel("Return grezzo")
    plt.title("Breakout: reward di training (media mobile 20 episodi)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "breakout_training_returns.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    for name, result in results.items():
        plt.plot(result["evaluation_steps"], result["evaluation_returns"], marker="o", label=labels[name])
        plt.axhline(float(result["random_baseline"]), color="gray", linestyle="--", alpha=0.35)
    plt.xlabel("Environment steps")
    plt.ylabel("Return medio in valutazione greedy")
    plt.title("Breakout: DQN e ablation vs policy casuale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "breakout_evaluation_comparison.png", dpi=150)
    plt.close()
    print(f"Grafici salvati in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
