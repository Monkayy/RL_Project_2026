from dataclasses import dataclass
from typing import Sequence, ClassVar
from utils.base_discretizer import BaseDiscretizer

from utils.config import (
    DEFAULT_BINS,
    DEFAULT_HIGH,
    DEFAULT_LOW
)

@dataclass(slots=True)
class CartPoleDiscretizer(BaseDiscretizer):
    """Converte un'osservazione continua di CartPole in un indice discreto."""

    bins_per_dimension: Sequence[int] = DEFAULT_BINS
    low: Sequence[float] = DEFAULT_LOW
    high: Sequence[float] = DEFAULT_HIGH

    expected_dim: ClassVar[int] = 4