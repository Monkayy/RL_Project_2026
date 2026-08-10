"""Funzioni per la creazione dei grafici sperimentali."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def smooth_runs(
    data: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Applica una media mobile separatamente
    a ogni seed.
    """

    data = np.asarray(
        data,
        dtype=np.float64,
    )

    if data.ndim != 2:
        raise ValueError(
            "data deve avere forma "
            "(numero_seed, numero_episodi)."
        )

    if window <= 1:
        return data.copy()

    if window > data.shape[1]:
        raise ValueError(
            "La finestra è maggiore del numero "
            "totale di episodi."
        )

    kernel = np.ones(
        window,
        dtype=np.float64,
    ) / window

    smoothed = [
        np.convolve(
            run,
            kernel,
            mode="valid",
        )
        for run in data
    ]

    return np.asarray(
        smoothed,
        dtype=np.float64,
    )


def plot_mean_and_std(
    results: dict[str, np.ndarray],
    title: str,
    ylabel: str,
    output_path: Path,
    window: int = 50,
) -> None:
    """
    Disegna la media e la deviazione standard
    delle curve ottenute sui seed.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(12, 7)
    )

    for label, data in results.items():
        smoothed = smooth_runs(
            data,
            window,
        )

        mean_values = np.mean(
            smoothed,
            axis=0,
        )

        std_values = np.std(
            smoothed,
            axis=0,
        )

        episodes = np.arange(
            window,
            window + len(mean_values),
        )

        plt.plot(
            episodes,
            mean_values,
            label=label,
            linewidth=2,
        )

        plt.fill_between(
            episodes,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.2,
        )

    plt.title(title)
    plt.xlabel("Episodio")
    plt.ylabel(ylabel)
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()

    print(
        f"Grafico salvato: {output_path}"
    )


def plot_bar_with_error(
    results: dict[str, np.ndarray],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Grafico a barre con media e deviazione standard.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = list(results.keys())

    means = [
        float(np.mean(results[label]))
        for label in labels
    ]

    standard_deviations = [
        float(np.std(results[label]))
        for label in labels
    ]

    positions = np.arange(
        len(labels)
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        positions,
        means,
        yerr=standard_deviations,
        capsize=6,
    )

    plt.xticks(
        positions,
        labels,
    )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(
        axis="y",
        alpha=0.3,
    )
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()

    print(
        f"Grafico salvato: {output_path}"
    )


def episodes_to_threshold(
    rewards: np.ndarray,
    threshold: float,
    window: int = 100,
) -> np.ndarray:
    """
    Calcola il primo episodio in cui la media mobile
    raggiunge una determinata soglia.
    """

    smoothed = smooth_runs(
        rewards,
        window,
    )

    episodes = []

    for run in smoothed:
        reached = np.flatnonzero(
            run >= threshold
        )

        if len(reached) == 0:
            episodes.append(
                np.nan
            )
        else:
            episodes.append(
                int(reached[0] + window)
            )

    return np.asarray(
        episodes,
        dtype=np.float64,
    )


def plot_speed_to_threshold(
    results: dict[str, np.ndarray],
    thresholds: list[float],
    title: str,
    output_path: Path,
    window: int = 100,
) -> None:
    """
    Confronta il numero di episodi necessario
    per raggiungere più soglie di reward.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    algorithm_names = list(
        results.keys()
    )

    x_positions = np.arange(
        len(thresholds)
    )

    total_width = 0.8

    bar_width = (
        total_width
        / len(algorithm_names)
    )

    plt.figure(
        figsize=(11, 7)
    )

    for algorithm_index, algorithm_name in enumerate(
        algorithm_names
    ):
        means = []
        standard_deviations = []

        for threshold in thresholds:
            episodes = episodes_to_threshold(
                results[algorithm_name],
                threshold,
                window,
            )

            reached = episodes[
                ~np.isnan(episodes)
            ]

            print(
                f"{algorithm_name}, soglia {threshold}: "
                f"{len(reached)}/{len(episodes)} seed "
                "hanno raggiunto la soglia."
            )

            if len(reached) == 0:
                means.append(
                    results[algorithm_name].shape[1]
                )
                standard_deviations.append(
                    0.0
                )
            else:
                means.append(
                    float(np.mean(reached))
                )
                standard_deviations.append(
                    float(np.std(reached))
                )

        offset = (
            algorithm_index
            - (len(algorithm_names) - 1) / 2
        ) * bar_width

        plt.bar(
            x_positions + offset,
            means,
            width=bar_width,
            yerr=standard_deviations,
            capsize=5,
            label=algorithm_name,
        )

    plt.xticks(
        x_positions,
        [
            str(threshold)
            for threshold in thresholds
        ],
    )

    plt.title(title)
    plt.xlabel("Soglia di reward")
    plt.ylabel("Episodi necessari")
    plt.grid(
        axis="y",
        alpha=0.3,
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()

    print(
        f"Grafico salvato: {output_path}"
    )


def plot_parameter_summary(
    parameter_values: list[float],
    final_scores: dict[str, list[float]],
    title: str,
    parameter_name: str,
    output_path: Path,
) -> None:
    """
    Riassume la reward finale media ottenuta
    per diversi valori di un iperparametro.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 6)
    )

    for algorithm_name, scores in final_scores.items():
        plt.plot(
            parameter_values,
            scores,
            marker="o",
            label=algorithm_name,
        )

    plt.title(title)
    plt.xlabel(parameter_name)
    plt.ylabel(
        "Reward media negli ultimi 100 episodi"
    )
    plt.grid(
        True,
        alpha=0.3,
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()

    print(
        f"Grafico salvato: {output_path}"
    )