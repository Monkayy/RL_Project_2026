"""Funzioni condivise per gli esperimenti multi-seed."""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from config import (
    DEFAULT_BINS,
    DEFAULT_HIGH,
    DEFAULT_LOW,
)

from cartpole.discretizer import (
    CartPoleDiscretizer,
)

from cartpole.results import (
    ControlResult,
    PredictionResult,
)


@dataclass(slots=True)
class MultiSeedControlResult:
    """Risultati di un algoritmo di controllo."""

    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    epsilons: np.ndarray
    q_tables: list[np.ndarray]


@dataclass(slots=True)
class MultiSeedPredictionResult:
    """Risultati di un algoritmo di prediction."""

    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    mean_absolute_updates: np.ndarray
    v_tables: list[np.ndarray]


def create_discretizer(
    bins_per_dimension=DEFAULT_BINS,
) -> CartPoleDiscretizer:
    """Crea il discretizzatore usato negli esperimenti."""

    return CartPoleDiscretizer(
        bins_per_dimension=bins_per_dimension,
        low=DEFAULT_LOW,
        high=DEFAULT_HIGH,
    )


def run_control_algorithm(
    algorithm: Callable[..., ControlResult],
    env_id: str,
    num_episodes: int,
    seeds: Sequence[int],
    algorithm_parameters: dict,
    bins_per_dimension=DEFAULT_BINS,
) -> MultiSeedControlResult:
    """
    Esegue Q-Learning o SARSA su più seed.
    """

    all_returns = []
    all_lengths = []
    all_epsilons = []
    q_tables = []

    for seed in seeds:
        print(
            f"  Esecuzione con seed {seed}..."
        )

        discretizer = create_discretizer(
            bins_per_dimension
        )

        result = algorithm(
            env_id=env_id,
            discretizer=discretizer,
            num_episodes=num_episodes,
            seed=seed,
            **algorithm_parameters,
        )

        all_returns.append(
            result.episode_returns
        )

        all_lengths.append(
            result.episode_lengths
        )

        all_epsilons.append(
            result.epsilons
        )

        q_tables.append(
            result.q_table
        )

    return MultiSeedControlResult(
        episode_returns=np.asarray(
            all_returns,
            dtype=np.float64,
        ),
        episode_lengths=np.asarray(
            all_lengths,
            dtype=np.float64,
        ),
        epsilons=np.asarray(
            all_epsilons,
            dtype=np.float64,
        ),
        q_tables=q_tables,
    )


def run_prediction_algorithm(
    algorithm: Callable[..., PredictionResult],
    env_id: str,
    num_episodes: int,
    seeds: Sequence[int],
    algorithm_parameters: dict,
    bins_per_dimension=DEFAULT_BINS,
) -> MultiSeedPredictionResult:
    """
    Esegue Monte Carlo Prediction o TD(0)
    su più seed.
    """

    all_returns = []
    all_lengths = []
    all_updates = []
    v_tables = []

    for seed in seeds:
        print(
            f"  Esecuzione con seed {seed}..."
        )

        discretizer = create_discretizer(
            bins_per_dimension
        )

        result = algorithm(
            env_id=env_id,
            discretizer=discretizer,
            num_episodes=num_episodes,
            seed=seed,
            **algorithm_parameters,
        )

        all_returns.append(
            result.episode_returns
        )

        all_lengths.append(
            result.episode_lengths
        )

        all_updates.append(
            result.mean_absolute_updates
        )

        v_tables.append(
            result.v_table
        )

    return MultiSeedPredictionResult(
        episode_returns=np.asarray(
            all_returns,
            dtype=np.float64,
        ),
        episode_lengths=np.asarray(
            all_lengths,
            dtype=np.float64,
        ),
        mean_absolute_updates=np.asarray(
            all_updates,
            dtype=np.float64,
        ),
        v_tables=v_tables,
    )