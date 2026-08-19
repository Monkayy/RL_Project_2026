"""Analisi di sensibilità agli iperparametri per Cartpole."""

import numpy as np

from utils.config import (
    ENV_ID,
    FIGURE_DIR,
    MONTE_CARLO_PARAMS,
    Q_LEARNING_PARAMS,
    SARSA_PARAMS,
    SEEDS,
    SENSITIVITY_EPISODES,
    TD_ZERO_PARAMS,
)

from algorithms.monte_carlo import monte_carlo_prediction
from algorithms.q_learning import q_learning
from algorithms.sarsa import sarsa
from algorithms.td_zero import td_zero_prediction

from experiments.plotting import plot_mean_and_std, plot_parameter_summary

from experiments.runners import run_control_algorithm, run_prediction_algorithm


def final_score(
    rewards: np.ndarray,
    final_window: int = 100,
) -> float:
    """
    Reward media negli ultimi episodi,
    prima mediata per seed.
    """

    number_of_episodes = rewards.shape[1]

    window = min(
        final_window,
        number_of_episodes,
    )

    per_seed_scores = np.mean(
        rewards[:, -window:],
        axis=1,
    )

    return float(
        np.mean(per_seed_scores)
    )


def run_control_parameter_study(
    algorithm_name: str,
    algorithm,
    base_parameters: dict,
    parameter_name: str,
    parameter_values: list[float],
    num_episodes: int,
) -> None:
    """Studia un parametro di Q-Learning o SARSA."""

    curves = {}
    summary_scores = []

    for value in parameter_values:
        parameters = dict(
            base_parameters
        )

        label = (
            f"{parameter_name}={value}"
        )

        if parameter_name == "epsilon":
            parameters[
                "epsilon_start"
            ] = value

            parameters[
                "epsilon_min"
            ] = value

            parameters[
                "epsilon_decay"
            ] = 1.0
        else:
            parameters[
                parameter_name
            ] = value

        print(
            f"\n{algorithm_name}: {label}"
        )

        results = run_control_algorithm(
            algorithm=algorithm,
            env_id=ENV_ID,
            num_episodes=num_episodes,
            seeds=SEEDS,
            algorithm_parameters=parameters,
        )

        curves[label] = (
            results.episode_returns
        )

        summary_scores.append(
            final_score(
                results.episode_returns
            )
        )

    plot_mean_and_std(
        results=curves,
        title=(
            f"{algorithm_name}: sensibilità "
            f"a {parameter_name}"
        ),
        ylabel="Reward totale",
        output_path=(
            FIGURE_DIR
            / (
                f"{algorithm_name.lower()}"
                f"_sensitivity_"
                f"{parameter_name}.png"
            )
        ),
        window=50,
    )

    plot_parameter_summary(
        parameter_values=parameter_values,
        final_scores={
            algorithm_name: summary_scores
        },
        title=(
            f"{algorithm_name}: risultato finale "
            f"rispetto a {parameter_name}"
        ),
        parameter_name=parameter_name,
        output_path=(
            FIGURE_DIR
            / (
                f"{algorithm_name.lower()}"
                f"_sensitivity_"
                f"{parameter_name}_summary.png"
            )
        ),
    )


def run_prediction_parameter_study(
    algorithm_name: str,
    algorithm,
    base_parameters: dict,
    parameter_name: str,
    parameter_values: list[float],
    num_episodes: int,
) -> None:
    """
    Studia alpha o gamma per Monte Carlo e TD(0).
    """

    curves = {}
    summary_scores = []

    for value in parameter_values:
        parameters = dict(
            base_parameters
        )

        parameters[
            parameter_name
        ] = value

        label = (
            f"{parameter_name}={value}"
        )

        print(
            f"\n{algorithm_name}: {label}"
        )

        results = run_prediction_algorithm(
            algorithm=algorithm,
            env_id=ENV_ID,
            num_episodes=num_episodes,
            seeds=SEEDS,
            algorithm_parameters=parameters,
        )

        curves[label] = (
            results.mean_absolute_updates
        )

        final_updates = (
            results.mean_absolute_updates[
                :,
                -min(
                    100,
                    num_episodes,
                ):
            ]
        )

        summary_scores.append(
            float(
                np.mean(final_updates)
            )
        )

    plot_mean_and_std(
        results=curves,
        title=(
            f"{algorithm_name}: sensibilità "
            f"a {parameter_name}"
        ),
        ylabel=(
            "Aggiornamento assoluto medio"
        ),
        output_path=(
            FIGURE_DIR
            / (
                f"{algorithm_name.lower()}"
                f"_sensitivity_"
                f"{parameter_name}.png"
            )
        ),
        window=50,
    )

    plot_parameter_summary(
        parameter_values=parameter_values,
        final_scores={
            algorithm_name: summary_scores
        },
        title=(
            f"{algorithm_name}: aggiornamento "
            f"finale rispetto a {parameter_name}"
        ),
        parameter_name=parameter_name,
        output_path=(
            FIGURE_DIR
            / (
                f"{algorithm_name.lower()}"
                f"_sensitivity_"
                f"{parameter_name}_summary.png"
            )
        ),
    )


def run_all_sensitivity_experiments(
    num_episodes: int = SENSITIVITY_EPISODES,
) -> None:
    """Esegue tutte le analisi previste."""

    alpha_control_values = [
        0.05,
        0.10,
        0.20,
        0.50,
    ]

    gamma_values = [
        0.80,
        0.90,
        0.99,
        1.00,
    ]

    epsilon_values = [
        0.01,
        0.05,
        0.10,
        0.30,
    ]

    alpha_prediction_values = [
        0.01,
        0.05,
        0.10,
        0.30,
    ]

    run_control_parameter_study(
        algorithm_name="Q-Learning",
        algorithm=q_learning,
        base_parameters=(
            Q_LEARNING_PARAMS
        ),
        parameter_name="alpha",
        parameter_values=(
            alpha_control_values
        ),
        num_episodes=num_episodes,
    )

    run_control_parameter_study(
        algorithm_name="Q-Learning",
        algorithm=q_learning,
        base_parameters=(
            Q_LEARNING_PARAMS
        ),
        parameter_name="gamma",
        parameter_values=gamma_values,
        num_episodes=num_episodes,
    )

    run_control_parameter_study(
        algorithm_name="Q-Learning",
        algorithm=q_learning,
        base_parameters=(
            Q_LEARNING_PARAMS
        ),
        parameter_name="epsilon",
        parameter_values=epsilon_values,
        num_episodes=num_episodes,
    )

    run_control_parameter_study(
        algorithm_name="SARSA",
        algorithm=sarsa,
        base_parameters=SARSA_PARAMS,
        parameter_name="alpha",
        parameter_values=(
            alpha_control_values
        ),
        num_episodes=num_episodes,
    )

    run_control_parameter_study(
        algorithm_name="SARSA",
        algorithm=sarsa,
        base_parameters=SARSA_PARAMS,
        parameter_name="gamma",
        parameter_values=gamma_values,
        num_episodes=num_episodes,
    )

    run_control_parameter_study(
        algorithm_name="SARSA",
        algorithm=sarsa,
        base_parameters=SARSA_PARAMS,
        parameter_name="epsilon",
        parameter_values=epsilon_values,
        num_episodes=num_episodes,
    )

    run_prediction_parameter_study(
        algorithm_name="Monte Carlo",
        algorithm=(
            monte_carlo_prediction
        ),
        base_parameters=(
            MONTE_CARLO_PARAMS
        ),
        parameter_name="alpha",
        parameter_values=(
            alpha_prediction_values
        ),
        num_episodes=num_episodes,
    )

    run_prediction_parameter_study(
        algorithm_name="Monte Carlo",
        algorithm=(
            monte_carlo_prediction
        ),
        base_parameters=(
            MONTE_CARLO_PARAMS
        ),
        parameter_name="gamma",
        parameter_values=gamma_values,
        num_episodes=num_episodes,
    )

    run_prediction_parameter_study(
        algorithm_name="TD(0)",
        algorithm=td_zero_prediction,
        base_parameters=TD_ZERO_PARAMS,
        parameter_name="alpha",
        parameter_values=(
            alpha_prediction_values
        ),
        num_episodes=num_episodes,
    )

    run_prediction_parameter_study(
        algorithm_name="TD(0)",
        algorithm=td_zero_prediction,
        base_parameters=TD_ZERO_PARAMS,
        parameter_name="gamma",
        parameter_values=gamma_values,
        num_episodes=num_episodes,
    )

    print(
        "\nAnalisi di sensibilità completata."
    )


if __name__ == "__main__":
    run_all_sensitivity_experiments()