"""TD(0) Prediction per CartPole."""

import gymnasium as gym
import numpy as np

from cartpole.discretizer import CartPoleDiscretizer
from cartpole.policies import (
    cartpole_heuristic_policy,
)
from cartpole.results import PredictionResult

from config import TD_ZERO_PARAMS as td0_params
default_alpha = td0_params.get("alpha")
default_gamma = td0_params.get("gamma")



def td_zero_prediction(
    env_id: str,
    discretizer: CartPoleDiscretizer,
    num_episodes: int,
    seed: int,
    alpha: float = default_alpha,
    gamma: float = default_gamma,
) -> PredictionResult:
    """Stima V(s) dopo ogni transizione."""

    if num_episodes < 1:
        raise ValueError(
            "num_episodes deve essere almeno 1."
        )

    if not 0.0 < alpha <= 1.0:
        raise ValueError(
            "alpha deve essere in (0, 1]."
        )

    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            "gamma deve essere in [0, 1]."
        )

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

        state = discretizer.encode(observation)
        finished = False
        absolute_updates = []

        while not finished:
            action = cartpole_heuristic_policy(
                observation
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            next_state = discretizer.encode(
                next_observation
            )

            finished = terminated or truncated

            if terminated:
                next_value = 0.0
            else:
                next_value = float(
                    v_table[next_state]
                )

            td_target = (
                float(reward)
                + gamma * next_value
            )

            old_value = v_table[state]

            td_error = td_target - old_value

            v_table[state] += alpha * td_error

            absolute_updates.append(
                abs(v_table[state] - old_value)
            )

            observation = next_observation
            state = next_state

            episode_returns[episode] += float(
                reward
            )

            episode_lengths[episode] += 1

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