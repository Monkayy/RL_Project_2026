"""Policy specifiche per Acrobot."""

from typing import Callable

import numpy as np

from acrobot.discretizer import AcrobotDiscretizer
from utils.policies import greedy_action

def make_greedy_policy(
    q_table: np.ndarray,
    discretizer: AcrobotDiscretizer,
    rng: np.random.Generator,
) -> Callable[[np.ndarray], int]:

    def policy(observation: np.ndarray) -> int:
        state = discretizer.encode(observation)
        return greedy_action(q_table[state], rng)

    return policy