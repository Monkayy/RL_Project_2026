"""Valutazione delle policy senza esplorazione."""

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

from cartpole.discretizer import (
    CartPoleDiscretizer,
)
from cartpole.policies import greedy_action


def evaluate_q_table(
    env_id: str,
    q_table: np.ndarray,
    discretizer: CartPoleDiscretizer,
    seeds: Sequence[int],
    episodes_per_seed: int = 20,
) -> np.ndarray:
    """Valuta una Q-table mediante policy greedy."""

    if episodes_per_seed < 1:
        raise ValueError(
            "episodes_per_seed deve essere almeno 1."
        )

    all_returns = []

    for seed in seeds:
        env = gym.make(env_id)
        env.action_space.seed(seed)

        rng = np.random.default_rng(seed)
        seed_returns = []

        for episode in range(episodes_per_seed):
            observation, _ = env.reset(
                seed=seed + episode
            )

            finished = False
            total_reward = 0.0

            while not finished:
                state = discretizer.encode(
                    observation
                )

                action = greedy_action(
                    q_table[state],
                    rng,
                )

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)

                finished = (
                    terminated or truncated
                )

                total_reward += float(reward)

            seed_returns.append(total_reward)

        env.close()
        all_returns.append(seed_returns)

    return np.asarray(
        all_returns,
        dtype=np.float64,
    )