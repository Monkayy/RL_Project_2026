"""Confronto Monte Carlo Prediction vs TD(0)."""

from pathlib import Path

import numpy as np

from config import (
    DATA_DIR,
    ENV_ID,
    FIGURE_DIR,
    MONTE_CARLO_PARAMS,
    PREDICTION_EPISODES,
    SEEDS,
    TD_ZERO_PARAMS,
)

from algorithms.monte_carlo import (
    monte_carlo_prediction,
)

from algorithms.td_zero import (
    td_zero_prediction,
)

from experiments.plotting import (
    plot_bar_with_error,
    plot_mean_and_std,
)

from experiments.runners import (
    run_prediction_algorithm,
)


def calculate_value_stability(
    v_tables: list[np.ndarray],
) -> np.ndarray:
    """
    Calcola la deviazione standard delle stime
    V(s) tra seed, considerando gli stati visitati.
    """

    stacked_tables = np.stack(
        v_tables,
        axis=0,
    )

    visited_mask = np.any(
        np.abs(stacked_tables) > 1e-12,
        axis=0,
    )

    if not np.any(visited_mask):
        return np.asarray(
            [0.0],
            dtype=np.float64,
        )

    state_standard_deviations = np.std(
        stacked_tables,
        axis=0,
    )

    return state_standard_deviations[
        visited_mask
    ]


def calculate_table_difference(
    first_tables: list[np.ndarray],
    second_tables: list[np.ndarray],
) -> float:
    """
    Calcola la differenza assoluta media tra
    le V-table medie dei due algoritmi.
    """

    first_mean = np.mean(
        np.stack(
            first_tables,
            axis=0,
        ),
        axis=0,
    )

    second_mean = np.mean(
        np.stack(
            second_tables,
            axis=0,
        ),
        axis=0,
    )

    visited_mask = (
        (np.abs(first_mean) > 1e-12)
        | (np.abs(second_mean) > 1e-12)
    )

    if not np.any(visited_mask):
        return 0.0

    return float(
        np.mean(
            np.abs(
                first_mean[visited_mask]
                - second_mean[visited_mask]
            )
        )
    )


def write_prediction_metrics(
    output_path: Path,
    mc_stability: np.ndarray,
    td_stability: np.ndarray,
    table_difference: float,
) -> None:
    """Salva le metriche testuali."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "Confronto Monte Carlo vs TD(0)\n"
        )

        file.write(
            "================================\n\n"
        )

        file.write(
            "Stabilità Monte Carlo "
            "(std media tra seed): "
            f"{np.mean(mc_stability):.6f}\n"
        )

        file.write(
            "Stabilità TD(0) "
            "(std media tra seed): "
            f"{np.mean(td_stability):.6f}\n"
        )

        file.write(
            "Differenza assoluta media "
            "tra V-table: "
            f"{table_difference:.6f}\n"
        )


def run_prediction_comparison(
    num_episodes: int = PREDICTION_EPISODES,
) -> None:
    """Esegue il confronto completo."""

    print(
        "\n======================================"
    )
    print(
        "CONFRONTO MONTE CARLO VS TD(0)"
    )
    print(
        "======================================"
    )

    print(
        "\nMonte Carlo Prediction"
    )

    monte_carlo_results = (
        run_prediction_algorithm(
            algorithm=monte_carlo_prediction,
            env_id=ENV_ID,
            num_episodes=num_episodes,
            seeds=SEEDS,
            algorithm_parameters=(
                MONTE_CARLO_PARAMS
            ),
        )
    )

    print(
        "\nTD(0) Prediction"
    )

    td_results = run_prediction_algorithm(
        algorithm=td_zero_prediction,
        env_id=ENV_ID,
        num_episodes=num_episodes,
        seeds=SEEDS,
        algorithm_parameters=TD_ZERO_PARAMS,
    )

    plot_mean_and_std(
        results={
            "Monte Carlo": (
                monte_carlo_results
                .mean_absolute_updates
            ),
            "TD(0)": (
                td_results
                .mean_absolute_updates
            ),
        },
        title=(
            "Monte Carlo vs TD(0): "
            "ampiezza degli aggiornamenti"
        ),
        ylabel=(
            "Aggiornamento assoluto medio"
        ),
        output_path=(
            FIGURE_DIR
            / "prediction_updates.png"
        ),
        window=50,
    )

    plot_mean_and_std(
        results={
            "Monte Carlo": (
                monte_carlo_results
                .episode_returns
            ),
            "TD(0)": (
                td_results
                .episode_returns
            ),
        },
        title=(
            "Reward della policy fissa"
        ),
        ylabel="Reward totale",
        output_path=(
            FIGURE_DIR
            / "prediction_rewards.png"
        ),
        window=50,
    )

    monte_carlo_stability = (
        calculate_value_stability(
            monte_carlo_results.v_tables
        )
    )

    td_stability = calculate_value_stability(
        td_results.v_tables
    )

    table_difference = (
        calculate_table_difference(
            monte_carlo_results.v_tables,
            td_results.v_tables,
        )
    )

    plot_bar_with_error(
        results={
            "Monte Carlo": (
                monte_carlo_stability
            ),
            "TD(0)": (
                td_stability
            ),
        },
        title=(
            "Variabilità delle stime V(s) "
            "tra seed"
        ),
        ylabel=(
            "Deviazione standard di V(s)"
        ),
        output_path=(
            FIGURE_DIR
            / "prediction_stability.png"
        ),
    )

    write_prediction_metrics(
        output_path=(
            DATA_DIR
            / "prediction_metrics.txt"
        ),
        mc_stability=(
            monte_carlo_stability
        ),
        td_stability=(
            td_stability
        ),
        table_difference=table_difference,
    )

    np.savez_compressed(
        DATA_DIR
        / "prediction_comparison_results.npz",
        monte_carlo_returns=(
            monte_carlo_results
            .episode_returns
        ),
        td_returns=(
            td_results
            .episode_returns
        ),
        monte_carlo_updates=(
            monte_carlo_results
            .mean_absolute_updates
        ),
        td_updates=(
            td_results
            .mean_absolute_updates
        ),
    )

    print(
        "\nStabilità Monte Carlo:",
        np.mean(monte_carlo_stability),
    )

    print(
        "Stabilità TD(0):",
        np.mean(td_stability),
    )

    print(
        "Differenza media tra V-table:",
        table_difference,
    )

    print(
        "\nConfronto prediction completato."
    )


if __name__ == "__main__":
    run_prediction_comparison()