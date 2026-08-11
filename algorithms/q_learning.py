"""Implementazione tabulare di Q-Learning."""

import gymnasium as gym
import numpy as np
from dataclasses import dataclass

from utils.base_discretizer import BaseDiscretizer
from utils.policies import (
    epsilon_greedy_action,
    update_epsilon,
)
from utils.results import ControlResult

from utils.config import Q_LEARNING_PARAMS as ql_params

default_alpha = ql_params.get("alpha")
default_gamma = ql_params.get("gamma")
default_eps_start = ql_params.get("epsilon_start")
default_eps_min = ql_params.get("epsilon_min")
default_eps_decay = ql_params.get("epsilon_decay")

@dataclass
class QLearningConfig:
    """Configurazione e validazione dei iperparametri per Q-Learning."""

    num_episodes: int
    alpha: float
    gamma: float
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float

    @classmethod
    def from_params(cls, num_episodes: int, params: dict) -> "QLearningConfig":
        """Costruisce la config a partire da un dict tipo *_PARAMS."""
        return cls(num_episodes=num_episodes, **params)

    def __post_init__(self) -> None:
        """Valida gli intervalli degli iperparametri dopo l'inizializzazione."""
        if self.num_episodes < 1:
            raise ValueError("num_episodes deve essere almeno 1.")

        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha deve essere in (0, 1].")

        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma deve essere in [0, 1].")

        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ValueError(
                "È richiesto 0 <= epsilon_min <= epsilon_start <= 1."
            )

        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay deve essere in (0, 1].")

        
def q_learning(
    env_id: str,
    discretizer: BaseDiscretizer,
    seed: int,
    config: QLearningConfig,
) -> ControlResult:
    """Addestra una policy con Q-Learning off-policy."""

    env = gym.make(env_id)
    env.action_space.seed(seed)

    rng = np.random.default_rng(seed)

    n_actions = int(env.action_space.n)

    q_table = np.zeros(
        (*discretizer.state_shape, n_actions),
        dtype=np.float64,
    )

    episode_returns = np.zeros(
        config.num_episodes,
        dtype=np.float64,
    )

    episode_lengths = np.zeros(
        config.num_episodes,
        dtype=np.int32,
    )

    epsilons = np.zeros(
        config.num_episodes,
        dtype=np.float64,
    )

    epsilon = config.epsilon_start

    for episode in range(config.num_episodes):
        reset_seed = seed if episode == 0 else None

        observation, _ = env.reset(seed=reset_seed)

        state = discretizer.encode(observation)
        finished = False

        while not finished:
            action = epsilon_greedy_action(
                q_table[state],
                epsilon,
                rng,
            )

            (
                next_observation,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            next_state = discretizer.encode(next_observation)

            finished = terminated or truncated

            if terminated:
                next_value = 0.0
            else:
                next_value = float(np.max(q_table[next_state]))

            target = float(reward) + config.gamma * next_value
            error = target - q_table[state][action]
            q_table[state][action] += config.alpha * error

            state = next_state

            episode_returns[episode] += float(reward)
            episode_lengths[episode] += 1

        epsilons[episode] = epsilon

        epsilon = update_epsilon(
            epsilon,
            config.epsilon_min,
            config.epsilon_decay,
        )

    env.close()

    return ControlResult(
        q_table=q_table,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        epsilons=epsilons,
    )