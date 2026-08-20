from abc import ABC
from dataclasses import dataclass, field
from typing import Sequence, ClassVar

import numpy as np


@dataclass(slots=True)
class BaseDiscretizer(ABC):
    bins_per_dimension: Sequence[int]
    low: Sequence[float]
    high: Sequence[float]

    expected_dim: ClassVar[int] = 0

    _low_arr: np.ndarray = field(init=False, repr=False)
    _high_arr: np.ndarray = field(init=False, repr=False)
    _boundaries: tuple[np.ndarray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not all(isinstance(n, int) for n in self.bins_per_dimension):
            raise TypeError("I bin devono essere interi.")

        if self.expected_dim > 0:
            if len(self.bins_per_dimension) != self.expected_dim:
                raise ValueError(
                    f"{self.__class__.__name__} richiede esattamente "
                    f"{self.expected_dim} dimensioni per i bin."
                )

            if not (len(self.low) == len(self.high) == self.expected_dim):
                raise ValueError(
                    f"I limiti low e high devono contenere esattamente {self.expected_dim} elementi."
                )

        if min(self.bins_per_dimension) < 1:
            raise ValueError("Ogni dimensione deve avere almeno 1 bin.")

        # Convert bounds to numpy arrays for vectorized operations
        self._low_arr = np.asarray(self.low, dtype=np.float64)
        self._high_arr = np.asarray(self.high, dtype=np.float64)

        if np.any(self._low_arr >= self._high_arr):
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
    def state_shape(self) -> tuple[int, ...]:
        """Returns the grid dimensions (bins count per dimension)."""
        return tuple(self.bins_per_dimension)

    def encode(self, observation: np.ndarray) -> tuple[int, ...]:
        """Maps continuous state observations to discrete bin indices."""
        values = np.asarray(observation, dtype=np.float64)

        if values.shape != (self.expected_dim,):
            raise ValueError(
                f"Osservazione {self.__class__.__name__} non valida: "
                f"shape {values.shape}, attesa ({self.expected_dim},)."
            )

        # Restrict values to [low, high] bounds to handle outliers
        clipped = np.clip(
            values,
            self._low_arr,
            self._high_arr,
        )

        # Map each scalar value to its corresponding bin index using pre-computed boundaries
        return tuple(
            int(np.digitize(value, boundaries))
            for value, boundaries in zip(
                clipped,
                self._boundaries,
            )
        )