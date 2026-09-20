"""Grafici comparativi per gli esperimenti DQN su Breakout."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from breakout.config import DATA_DIR, FIGURE_DIR

LABELS = {
    "full": "DQN completo",
    "no_target": "senza target network",
    "no_replay": "senza experience replay",
    "neither": "senza entrambe",
    "ddqn": "+ Double DQN",
    "dueling": "+ Dueling",
    "per": "+ PER",
    "bonus_full": "+ Double + Dueling + PER",
}
GROUPS = {
    "ablation": ("full", "no_target", "no_replay", "neither"),
    "bonus": ("full", "ddqn", "dueling", "per", "bonus_full"),
}


def moving_mean(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) == 0:
        return values
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def load_results(seed: int):
    results = {}
    for name in LABELS:
        path = DATA_DIR / f"{name}_seed_{seed}.npz"
        if path.exists():
            results[name] = np.load(path, allow_pickle=True)
    if not results:
        raise FileNotFoundError("Nessun risultato trovato: esegui prima breakout.run_experiments.")
    return results


def plot_group(results, group: str, names: tuple[str, ...]) -> list[Path]:
    available = [n for n in names if n in results]
    if len(available) < 1:  # niente da confrontare
        print(f"[{group}] saltato: serve almeno 1 esperimento, trovati {available}")
        return []

    saved = []

    plt.figure(figsize=(9, 5))
    for name in available:
        curve = moving_mean(results[name]["episode_returns"])
        if len(curve):
            plt.plot(np.arange(len(curve)), curve, label=LABELS[name])
    plt.xlabel("Episodio")
    plt.ylabel("Return grezzo")
    plt.title(f"Breakout ({group}): reward di training (media mobile 20 episodi)")
    plt.legend()
    plt.tight_layout()
    path = FIGURE_DIR / f"breakout_{group}_training_returns.png"
    plt.savefig(path, dpi=150)
    plt.close()
    saved.append(path)

    plt.figure(figsize=(9, 5))
    for name in available:
        r = results[name]
        plt.plot(r["evaluation_steps"], r["evaluation_returns"], marker="o", label=LABELS[name])
    plt.axhline(float(results[available[0]]["random_baseline"]), color="gray", linestyle="--", alpha=0.35)
    plt.xlabel("Environment steps")
    plt.ylabel("Return medio in valutazione greedy")
    plt.title(f"Breakout ({group}) vs policy casuale")
    plt.legend()
    plt.tight_layout()
    path = FIGURE_DIR / f"breakout_{group}_evaluation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    saved.append(path)

    print(f"[{group}] esperimenti: {', '.join(available)}")
    for p in saved:
        print(f"  salvato: {p}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = load_results(args.seed)
    print(f"Risultati caricati (seed {args.seed}): {', '.join(results)}")
    total = sum(len(plot_group(results, group, names)) for group, names in GROUPS.items())
    print(f"Totale grafici: {total} in {FIGURE_DIR}")

if __name__ == "__main__":
    main()