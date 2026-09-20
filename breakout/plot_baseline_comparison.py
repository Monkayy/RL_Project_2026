import argparse

import matplotlib.pyplot as plt
import numpy as np

from breakout.config import DATA_DIR, FIGURE_DIR


METHODS = {
    "full": ("Implementazione DQN", "tab:blue"),
    "baseline": ("Stable-Baselines3 DQN", "tab:orange"),
}


def paired_seeds(requested: list[int] | None) -> list[int]:
    if requested is not None:
        candidates = requested
    else:
        candidates = sorted(
            int(path.stem.rsplit("_", 1)[1])
            for path in DATA_DIR.glob("full_seed_*.npz")
        )
    return [
        seed for seed in candidates
        if (DATA_DIR / f"full_seed_{seed}.npz").exists()
        and (DATA_DIR / f"baseline_seed_{seed}.npz").exists()
    ]


def load_method(method: str, seeds: list[int]) -> list[np.lib.npyio.NpzFile]:
    return [np.load(DATA_DIR / f"{method}_seed_{seed}.npz", allow_pickle=True) for seed in seeds]


def aligned_evaluations(runs: list[np.lib.npyio.NpzFile]) -> tuple[np.ndarray, np.ndarray]:
    """Return common checkpoints and one evaluation curve per paired seed."""
    common = set(runs[0]["evaluation_steps"].tolist())
    for run in runs[1:]:
        common.intersection_update(run["evaluation_steps"].tolist())
    steps = np.asarray(sorted(common), dtype=np.int32)
    if not len(steps):
        raise ValueError("I run non hanno checkpoint di valutazione in comune.")
    curves = []
    for run in runs:
        lookup = dict(zip(run["evaluation_steps"].tolist(), run["evaluation_returns"].tolist()))
        curves.append([lookup[int(step)] for step in steps])
    return steps, np.asarray(curves, dtype=np.float64)


def mean_and_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seed da includere; default: tutte le coppie disponibili")
    parser.add_argument("--last-checkpoints", type=int, default=3,
                        help="Checkpoint finali usati nella barra di performance")
    args = parser.parse_args()
    if args.last_checkpoints < 1:
        raise ValueError("--last-checkpoints deve essere positivo")

    seeds = paired_seeds(args.seeds)
    if not seeds:
        raise FileNotFoundError(
            "Nessuna coppia trovata. Esegui sia run_ablation --only full sia run_baseline con lo stesso seed."
        )

    data = {method: load_method(method, seeds) for method in METHODS}
    curves = {method: aligned_evaluations(runs) for method, runs in data.items()}
    reference_steps = curves["full"][0]
    if not np.array_equal(reference_steps, curves["baseline"][0]):
        raise ValueError("I metodi non hanno checkpoint comuni identici.")

    suffix = f" ({len(seeds)} seed" + ("s)" if len(seeds) != 1 else ")")
    plt.figure(figsize=(9, 5))
    for method, (label, color) in METHODS.items():
        steps, values = curves[method]
        mean, std = mean_and_std(values)
        plt.plot(steps, mean, marker="o", label=label, color=color)
        if len(seeds) > 1:
            plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.20,
                             label=f"{label}: deviazione standard")
    plt.xlabel("Environment steps")
    plt.ylabel("Return medio: policy deterministica")
    plt.title("Breakout: DQN custom vs Stable-Baselines3" + suffix)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "breakout_sb3_evaluation_comparison.png", dpi=180)
    plt.close()

    labels, final_means, final_stds, auc_means, auc_stds = [], [], [], [], []
    for method, (label, _) in METHODS.items():
        _, values = curves[method]
        tail = values[:, -min(args.last_checkpoints, values.shape[1]):].mean(axis=1)
        # A 25k-step smoke run has only one evaluation point.  In that case
        # its only score is the meaningful summary; normal AUC needs >=2.
        if len(reference_steps) == 1:
            auc = values[:, 0]
        else:
            auc = np.trapz(values, x=reference_steps, axis=1) / (reference_steps[-1] - reference_steps[0])
        labels.append(label)
        final_means.append(tail.mean())
        final_stds.append(tail.std(ddof=1) if len(tail) > 1 else 0.0)
        auc_means.append(auc.mean())
        auc_stds.append(auc.std(ddof=1) if len(auc) > 1 else 0.0)

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].bar(x, final_means, yerr=final_stds, capsize=5,
                color=[METHODS[m][1] for m in METHODS])
    axes[0].set_xticks(x, labels, rotation=12, ha="right")
    axes[0].set_ylabel(f"Media ultimi {min(args.last_checkpoints, len(reference_steps))} checkpoint")
    axes[0].set_title("Performance finale")
    axes[1].bar(x, auc_means, yerr=auc_stds, capsize=5,
                color=[METHODS[m][1] for m in METHODS])
    axes[1].set_xticks(x, labels, rotation=12, ha="right")
    axes[1].set_ylabel("AUC normalizzata (return)" if len(reference_steps) > 1 else "Return al solo checkpoint")
    axes[1].set_title("Efficienza di apprendimento" if len(reference_steps) > 1 else "Checkpoint disponibile")
    fig.suptitle("Breakout: confronto paired custom DQN / SB3" + suffix)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "breakout_sb3_summary.png", dpi=180)
    plt.close(fig)

    print(f"Seed paired inclusi: {seeds}")
    print(f"Grafici salvati in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
