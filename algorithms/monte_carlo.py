"""Monte Carlo Prediction per CartPole."""
from typing import Callable

import gymnasium as gym
import numpy as np

from utils.base_discretizer import BaseDiscretizer

from utils.results import PredictionResult

from utils.config import MONTE_CARLO_PARAMS as mc_params
default_alpha = mc_params.get("alpha")
default_gamma = mc_params.get("gamma")

def monte_carlo_prediction(
    env_id: str,
    discretizer: BaseDiscretizer,
    policy: Callable[[np.ndarray], int],
    num_episodes: int,
    seed: int,
    alpha: float = default_alpha,
    gamma: float = default_gamma,
    first_visit: bool = True,
) -> PredictionResult:
    """Stima V(s) usando il ritorno completo."""

    if num_episodes < 1:
        raise ValueError(
            "num_episodes deve essere almeno 1."
        )

    if not 0.0 < alpha <= 1.0:
        raise ValueError(
            "alpha deve essere nell'intervallo (0, 1]."
        )

    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            "gamma deve essere nell'intervallo [0, 1]."
        )

    # creates the CartPole environment via Gymnasium
    env = gym.make(env_id)
    env.action_space.seed(seed)

    v_table = np.zeros(
        discretizer.state_shape,
        dtype=np.float64,
    )

    episode_returns = np.zeros(
        num_episodes,
        dtype=np.float64,
    )

    episode_lengths = np.zeros(
        num_episodes,
        dtype=np.int32,
    )

    mean_absolute_updates = np.zeros(
        num_episodes,
        dtype=np.float64,
    )

    for episode in range(num_episodes):
        reset_seed = seed if episode == 0 else None

        observation, _ = env.reset(
            seed=reset_seed
        )

        trajectory = []
        finished = False

        while not finished:
            state = discretizer.encode(observation)

            action = policy(observation)

            (
                observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            finished = terminated or truncated

            trajectory.append(
                (state, float(reward))
            )

        episode_returns[episode] = sum(
            reward
            for _, reward in trajectory
        )

        episode_lengths[episode] = len(trajectory)

        returns_from_t = np.zeros(
            len(trajectory),
            dtype=np.float64,
        )

        discounted_return = 0.0

        for idx in range(
            len(trajectory) - 1,
            -1,
            -1,
        ):
            discounted_return = (
                trajectory[idx][1]
                + gamma * discounted_return
            )

            returns_from_t[idx] = (
                discounted_return
            )

        visited = set()
        absolute_updates = []

        for idx, (state, _) in enumerate(
            trajectory
        ):
            if first_visit and state in visited:
                continue

            visited.add(state)

            old_value = v_table[state]

            v_table[state] += alpha * (
                returns_from_t[idx]
                - old_value
            )

            absolute_updates.append(
                abs(v_table[state] - old_value)
            )

        if absolute_updates:
            mean_absolute_updates[episode] = (
                float(np.mean(absolute_updates))
            )

    env.close()

    return PredictionResult(
        v_table=v_table,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        mean_absolute_updates=mean_absolute_updates,
    )