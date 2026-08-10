"""Discretizzazione dello spazio continuo di CartPole."""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class CartPoleDiscretizer:
    """Converte un'osservazione continua in un indice discreto."""

    bins_per_dimension: Sequence[int] = (8, 8, 12, 12)
    low: Sequence[float] = (-2.4, -3.0, -0.2095, -4.0)
    high: Sequence[float] = (2.4, 3.0, 0.2095, 4.0)

    _boundaries: tuple[np.ndarray, ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not (
            len(self.bins_per_dimension)
            == len(self.low)
            == len(self.high)
            == 4
        ):
            raise ValueError(
                "CartPole richiede esattamente quattro dimensioni."
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
            int(value)
            for value in self.bins_per_dimension
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
    def state_shape(self) -> tuple[int, int, int, int]:
        return tuple(self.bins_per_dimension)

    def encode(
        self,
        observation: np.ndarray,
    ) -> tuple[int, int, int, int]:

        values = np.asarray(
            observation,
            dtype=np.float64,
        )

        if values.shape != (4,):
            raise ValueError(
                f"Osservazione CartPole non valida: "
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