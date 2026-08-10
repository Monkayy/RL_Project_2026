"""Discretizzazione dello spazio continuo di Acrobot."""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from config import ACROBOT_DEFAULT_BINS, ACROBOT_DEFAULT_HIGH, ACROBOT_DEFAULT_LOW


@dataclass(slots=True)
class AcrobotDiscretizer:
    """Converte un'osservazione continua di Acrobot in un indice discreto.

    L'osservazione di Acrobot-v1 e' un vettore a sei dimensioni:
    [cos(theta1), sin(theta1), cos(theta2), sin(theta2),
     velocita' angolare di theta1, velocita' angolare di theta2].
    """

    bins_per_dimension: Sequence[int] = ACROBOT_DEFAULT_BINS
    low: Sequence[float] = ACROBOT_DEFAULT_LOW
    high: Sequence[float] = ACROBOT_DEFAULT_HIGH

    _boundaries: tuple[np.ndarray, ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not (
            len(self.bins_per_dimension)
            == len(self.low)
            == len(self.high)
            == 6
        ):
            raise ValueError(
                "Acrobot richiede esattamente sei dimensioni."
            )

        if any(
            int(n_bins) < 1
            for n_bins in self.bins_per_dimension
        ):
            raise ValueError(
                "Ogni dimensione deve avere almeno un bin."
            )

        low = np.asarray(self.low, dtype=np.float64)
        high = np.asarray(self.high, dtype=np.float64)
        bins = tuple(
            int(value) for value in self.bins_per_dimension
        )

        if np.any(low >= high):
            raise ValueError(
                "Ogni limite inferiore deve essere minore del superiore."
            )

        self.bins_per_dimension = bins
        self.low = tuple(float(value) for value in low)
        self.high = tuple(float(value) for value in high)

        # Per creare n intervalli servono n - 1 soglie interne.
        self._boundaries = tuple(
            np.linspace(
                low_value,
                high_value,
                n_bins + 1,
                dtype=np.float64,
            )[1:-1]
            for low_value, high_value, n_bins in zip(
                low,
                high,
                bins,
            )
        )

    @property
    def state_shape(
        self,
    ) -> tuple[int, int, int, int, int, int]:
        return tuple(self.bins_per_dimension)

    def encode(
        self,
        observation: np.ndarray,
    ) -> tuple[int, int, int, int, int, int]:
        values = np.asarray(
            observation,
            dtype=np.float64,
        )

        if values.shape != (6,):
            raise ValueError(
                f"Osservazione Acrobot non valida: "
                f"shape {values.shape}."
            )

        clipped = np.clip(
            values,
            np.asarray(self.low),
            np.asarray(self.high),
        )

        state = tuple(
            int(np.digitize(value, boundaries))
            for value, boundaries in zip(
                clipped,
                self._boundaries,
            )
        )

        return state