"""Failure analysis: discretizzazione troppo grossolana."""

import numpy as np

from utils.config import (
    DATA_DIR,
    ENV_ID,
    FIGURE_DIR,
    Q_LEARNING_PARAMS,
    SEEDS,
)

from algorithms.q_learning import (
    q_learning,
)

from cartpole.evaluation import (
    evaluate_q_table,
)

from experiments.plotting import (
    plot_bar_with_error,
    plot_mean_and_std,
)

from experiments.runners import (
    create_discretizer,
    run_control_algorithm,
)


def evaluate_failure_tables(
    q_tables: list[np.ndarray],
    bins_per_dimension,
) -> np.ndarray:
    """Valuta le Q-table del failure case."""

    returns = []

    discretizer = create_discretizer(
        bins_per_dimension
    )

    for index, q_table in enumerate(
        q_tables
    ):
        evaluation = evaluate_q_table(
            env_id=ENV_ID,
            q_table=q_table,
            discretizer=discretizer,
            seeds=(
                5000 + index,
                6000 + index,
                7000 + index,
            ),
            episodes_per_seed=20,
        )

        returns.extend(
            evaluation.flatten()
        )

    return np.asarray(
        returns,
        dtype=np.float64,
    )


def run_failure_case(
    num_episodes: int = 2000,
) -> None:
    """
    Confronta una discretizzazione normale con
    una discretizzazione costituita da un solo
    intervallo per ogni variabile.
    """

    print(
        "\n======================================"
    )
    print(
        "FAILURE CASE: DISCRETIZZAZIONE"
    )
    print(
        "======================================"
    )

    normal_bins = (
        8,
        8,
        12,
        12,
    )

    extremely_coarse_bins = (
        3,#1,
        3,#1,
        3,#1,
        3,#1,
    )

    print(
        "\nDiscretizzazione normale..."
    )

    normal_results = run_control_algorithm(
        algorithm=q_learning,
        env_id=ENV_ID,
        num_episodes=num_episodes,
        seeds=SEEDS,
        algorithm_parameters=(
            Q_LEARNING_PARAMS
        ),
        bins_per_dimension=normal_bins,
    )

    print(
        "\nDiscretizzazione estremamente "
        "grossolana..."
    )

    coarse_results = run_control_algorithm(
        algorithm=q_learning,
        env_id=ENV_ID,
        num_episodes=num_episodes,
        seeds=SEEDS,
        algorithm_parameters=(
            Q_LEARNING_PARAMS
        ),
        bins_per_dimension=(
            extremely_coarse_bins
        ),
    )

    plot_mean_and_std(
        results={
            "Discretizzazione normale": (
                normal_results
                .episode_returns
            ),
            "Un solo stato discreto": (
                coarse_results
                .episode_returns
            ),
        },
        title=(
            "Failure case: effetto della "
            "discretizzazione"
        ),
        ylabel="Reward totale",
        output_path=(
            FIGURE_DIR
            / "failure_discretization.png"
        ),
        window=50,
    )

    normal_evaluation = (
        evaluate_failure_tables(
            normal_results.q_tables,
            normal_bins,
        )
    )

    coarse_evaluation = (
        evaluate_failure_tables(
            coarse_results.q_tables,
            extremely_coarse_bins,
        )
    )

    plot_bar_with_error(
        results={
            "Normale": normal_evaluation,
            "Un solo stato": (
                coarse_evaluation
            ),
        },
        title=(
            "Valutazione del failure case"
        ),
        ylabel="Reward finale",
        output_path=(
            FIGURE_DIR
            / "failure_evaluation.png"
        ),
    )

    np.savez_compressed(
        DATA_DIR
        / "failure_case_results.npz",
        normal_training_returns=(
            normal_results
            .episode_returns
        ),
        coarse_training_returns=(
            coarse_results
            .episode_returns
        ),
        normal_evaluation=(
            normal_evaluation
        ),
        coarse_evaluation=(
            coarse_evaluation
        ),
    )

    print(
        "\nReward medio discretizzazione normale:",
        np.mean(normal_evaluation),
    )

    print(
        "Reward medio con un solo stato:",
        np.mean(coarse_evaluation),
    )

    print(
        "\nFailure case completato."
    )


if __name__ == "__main__":
    run_failure_case()