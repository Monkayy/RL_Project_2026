"""Avvio degli esperimenti del progetto CartPole."""

import argparse

from experiments.control_comparison import run_control_comparison
from experiments.failure_case import run_failure_case
from experiments.prediction_comparison import run_prediction_comparison
from experiments.sensitivity import run_all_sensitivity_experiments


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Esperimenti di Reinforcement "
            "Learning su CartPole."
        )
    )

    parser.add_argument(
        "experiment",
        choices=(
            "control",
            "prediction",
            "sensitivity",
            "failure",
            "all",
        ),
        help=(
            "Esperimento da eseguire."
        ),
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Sovrascrive il numero di episodi "
            "impostato nella configurazione."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if (
        arguments.episodes is not None
        and arguments.episodes < 1
    ):
        raise ValueError(
            "Il numero di episodi deve "
            "essere positivo."
        )

    if arguments.experiment == "control":
        if arguments.episodes is None:
            run_control_comparison()
        else:
            run_control_comparison(
                arguments.episodes
            )

    elif arguments.experiment == "prediction":
        if arguments.episodes is None:
            run_prediction_comparison()
        else:
            run_prediction_comparison(
                arguments.episodes
            )

    elif arguments.experiment == "sensitivity":
        if arguments.episodes is None:
            run_all_sensitivity_experiments()
        else:
            run_all_sensitivity_experiments(
                arguments.episodes
            )

    elif arguments.experiment == "failure":
        if arguments.episodes is None:
            run_failure_case()
        else:
            run_failure_case(
                arguments.episodes
            )

    elif arguments.experiment == "all":
        if arguments.episodes is None:
            run_control_comparison()
            run_prediction_comparison()
            run_all_sensitivity_experiments()
            run_failure_case()
        else:
            run_control_comparison(
                arguments.episodes
            )

            run_prediction_comparison(
                arguments.episodes
            )

            run_all_sensitivity_experiments(
                arguments.episodes
            )

            run_failure_case(
                arguments.episodes
            )


if __name__ == "__main__":
    main()