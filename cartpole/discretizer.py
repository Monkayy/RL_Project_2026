"""Discretizzazione dello spazio continuo di CartPole."""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from config import (DEFAULT_LOW, DEFAULT_HIGH, DEFAULT_BINS)


@dataclass(slots=True)
class CartPoleDiscretizer:
    """Converte un'osservazione continua in un indice discreto."""

    # (cart_pos, cart_vel, pole_angle, pole_angle_vel)
    bins_per_dimension: Sequence[int] = DEFAULT_BINS
    low: Sequence[float] = DEFAULT_LOW
    high: Sequence[float] = DEFAULT_HIGH

    low_arr = np.asarray(low, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)

    _boundaries: tuple[np.ndarray, ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not all(isinstance(n, int) for n in self.bins_per_dimension):
            raise TypeError("I bin devono essere interi.")

        if len(self.bins_per_dimension) != 4:
            raise ValueError("CartPole richiede esattamente quattro dimensioni.")
        
        if not (len(self.low) == len(self.high) == 4):
            raise ValueError("I limiti low e high devono contenere esattamente 4 elementi.")
        
        if min(self.bins_per_dimension) < 1:
            raise ValueError("Ogni dimensione deve avere almeno 1 bin.")

        if np.any(self.low_arr >= self.high_arr):
            raise ValueError(
                "Ogni limite inferiore deve essere minore del superiore."
            )

        # Per creare n intervalli servono n - 1 soglie interne.
        self._boundaries = tuple(
            np.linspace(
                low_value,
                high_value,
                n_bins + 1,
                dtype=np.float64,
            )[1:-1]
            for low_value, high_value, n_bins in zip(
                self.low,
                self.high,
                self.bins_per_dimension,
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
            self.low_arr,
            self.high_arr,
        )

        state = tuple(
            int(np.digitize(value, boundaries))
            for value, boundaries in zip(
                clipped,
                self._boundaries,
            )
        )

        return state