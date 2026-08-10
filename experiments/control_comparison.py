"""Confronto sperimentale tra Q-Learning e SARSA."""

import numpy as np

from config import (
    CONTROL_EPISODES,
    DATA_DIR,
    ENV_ID,
    FIGURE_DIR,
    Q_LEARNING_PARAMS,
    SARSA_PARAMS,
    SEEDS,
)

from cartpole.algorithms.q_learning import (
    q_learning,
)

from cartpole.algorithms.sarsa import (
    sarsa,
)

from cartpole.evaluation import (
    evaluate_q_table,
)

from experiments.plotting import (
    plot_bar_with_error,
    plot_mean_and_std,
    plot_speed_to_threshold,
)

from experiments.runners import (
    create_discretizer,
    run_control_algorithm,
)


def evaluate_all_q_tables(
    q_tables: list[np.ndarray],
) -> np.ndarray:
    """
    Valuta tutte le Q-table mediante policy greedy,
    senza esplorazione.
    """

    evaluation_returns = []

    discretizer = create_discretizer()

    evaluation_seeds = (
        1000,
        2000,
        3000,
    )

    for q_table in q_tables:
        returns = evaluate_q_table(
            env_id=ENV_ID,
            q_table=q_table,
            discretizer=discretizer,
            seeds=evaluation_seeds,
            episodes_per_seed=20,
        )

        evaluation_returns.extend(
            returns.flatten()
        )

    return np.asarray(
        evaluation_returns,
        dtype=np.float64,
    )


def run_control_comparison(
    num_episodes: int = CONTROL_EPISODES,
) -> None:
    """Esegue il confronto completo."""

    print(
        "\n======================================"
    )
    print(
        "CONFRONTO Q-LEARNING VS SARSA"
    )
    print(
        "======================================"
    )

    print(
        "\nAddestramento Q-Learning"
    )

    q_learning_results = run_control_algorithm(
        algorithm=q_learning,
        env_id=ENV_ID,
        num_episodes=num_episodes,
        seeds=SEEDS,
        algorithm_parameters=Q_LEARNING_PARAMS,
    )

    print(
        "\nAddestramento SARSA"
    )

    sarsa_results = run_control_algorithm(
        algorithm=sarsa,
        env_id=ENV_ID,
        num_episodes=num_episodes,
        seeds=SEEDS,
        algorithm_parameters=SARSA_PARAMS,
    )

    plot_mean_and_std(
        results={
            "Q-Learning": (
                q_learning_results.episode_returns
            ),
            "SARSA": (
                sarsa_results.episode_returns
            ),
        },
        title=(
            "CartPole: Q-Learning vs SARSA"
        ),
        ylabel="Reward totale",
        output_path=(
            FIGURE_DIR
            / "control_learning_curves.png"
        ),
        window=50,
    )

    plot_mean_and_std(
        results={
            "Q-Learning": (
                q_learning_results.episode_lengths
            ),
            "SARSA": (
                sarsa_results.episode_lengths
            ),
        },
        title=(
            "CartPole: durata degli episodi"
        ),
        ylabel="Numero di step",
        output_path=(
            FIGURE_DIR
            / "control_episode_lengths.png"
        ),
        window=50,
    )

    plot_speed_to_threshold(
        results={
            "Q-Learning": (
                q_learning_results.episode_returns
            ),
            "SARSA": (
                sarsa_results.episode_returns
            ),
        },
        thresholds=[
            100,
            200,
            400,
        ],
        title=(
            "Velocità di apprendimento"
        ),
        output_path=(
            FIGURE_DIR
            / "control_learning_speed.png"
        ),
        window=100,
    )

    print(
        "\nValutazione finale Q-Learning..."
    )

    q_learning_evaluation = evaluate_all_q_tables(
        q_learning_results.q_tables
    )

    print(
        "\nValutazione finale SARSA..."
    )

    sarsa_evaluation = evaluate_all_q_tables(
        sarsa_results.q_tables
    )

    plot_bar_with_error(
        results={
            "Q-Learning": (
                q_learning_evaluation
            ),
            "SARSA": (
                sarsa_evaluation
            ),
        },
        title=(
            "Valutazione delle policy greedy"
        ),
        ylabel="Reward dell'episodio",
        output_path=(
            FIGURE_DIR
            / "control_final_evaluation.png"
        ),
    )

    np.savez_compressed(
        DATA_DIR
        / "control_comparison_results.npz",
        q_learning_returns=(
            q_learning_results.episode_returns
        ),
        sarsa_returns=(
            sarsa_results.episode_returns
        ),
        q_learning_lengths=(
            q_learning_results.episode_lengths
        ),
        sarsa_lengths=(
            sarsa_results.episode_lengths
        ),
        q_learning_evaluation=(
            q_learning_evaluation
        ),
        sarsa_evaluation=(
            sarsa_evaluation
        ),
    )

    print(
        "\nReward medio finale Q-Learning:",
        np.mean(q_learning_evaluation),
    )

    print(
        "Deviazione standard Q-Learning:",
        np.std(q_learning_evaluation),
    )

    print(
        "\nReward medio finale SARSA:",
        np.mean(sarsa_evaluation),
    )

    print(
        "Deviazione standard SARSA:",
        np.std(sarsa_evaluation),
    )

    print(
        "\nConfronto completato."
    )


if __name__ == "__main__":
    run_control_comparison()