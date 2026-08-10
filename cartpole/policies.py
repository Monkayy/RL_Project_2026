"""Policy condivise dagli algoritmi di Reinforcement Learning."""

import numpy as np


def greedy_action(
    q_values: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """
    Sceglie un'azione greedy.

    Se più azioni hanno lo stesso valore Q massimo,
    ne seleziona casualmente una.
    """

    q_values = np.asarray(q_values, dtype=np.float64)

    if q_values.ndim != 1:
        raise ValueError(
            "q_values deve essere un array monodimensionale."
        )

    if len(q_values) == 0:
        raise ValueError(
            "q_values non può essere vuoto."
        )

    maximum = np.max(q_values)

    best_actions = np.flatnonzero(
        np.isclose(q_values, maximum)
    )

    return int(rng.choice(best_actions))


def epsilon_greedy_action(
    q_values: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """
    Sceglie un'azione mediante strategia epsilon-greedy.

    Con probabilità epsilon sceglie un'azione casuale.
    Altrimenti sceglie una delle migliori azioni.
    """

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(
            "epsilon deve essere compreso tra 0 e 1."
        )

    q_values = np.asarray(q_values, dtype=np.float64)

    if q_values.ndim != 1:
        raise ValueError(
            "q_values deve essere un array monodimensionale."
        )

    if len(q_values) == 0:
        raise ValueError(
            "q_values non può essere vuoto."
        )

    if rng.random() < epsilon:
        return int(
            rng.integers(
                low=0,
                high=len(q_values),
            )
        )

    return greedy_action(
        q_values=q_values,
        rng=rng,
    )


def update_epsilon(
    current: float,
    minimum: float,
    decay: float,
) -> float:
    """
    Riduce epsilon usando un decadimento moltiplicativo.

    Il valore non può scendere sotto il minimo specificato.
    """

    if not 0.0 <= current <= 1.0:
        raise ValueError(
            "current deve essere compreso tra 0 e 1."
        )

    if not 0.0 <= minimum <= 1.0:
        raise ValueError(
            "minimum deve essere compreso tra 0 e 1."
        )

    if minimum > current:
        raise ValueError(
            "minimum non può essere maggiore di current."
        )

    if not 0.0 < decay <= 1.0:
        raise ValueError(
            "decay deve essere compreso tra 0 escluso e 1."
        )

    return max(
        minimum,
        current * decay,
    )


def cartpole_heuristic_policy(
    observation: np.ndarray,
) -> int:
    """
    Policy euristica fissa per Monte Carlo Prediction e TD(0).

    L'azione viene scelta considerando principalmente:
     - angolo del palo;
     - velocità angolare del palo;
     - velocità del carrello.

    Azioni CartPole:
     - 0 = spinta verso sinistra
     - 1 = spinta verso destra
    """

    observation = np.asarray(
        observation,
        dtype=np.float64,
    )

    if observation.shape != (4,):
        raise ValueError(
            "L'osservazione di CartPole deve contenere "
            "esattamente quattro valori."
        )

    (
        _cart_position,
        cart_velocity,
        pole_angle,
        pole_angular_velocity,
    ) = observation

    score = (
        pole_angle
        + 0.18 * pole_angular_velocity
        + 0.015 * cart_velocity
    )

    return 1 if score >= 0.0 else 0