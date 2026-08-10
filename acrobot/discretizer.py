from dataclasses import dataclass
from typing import Sequence, ClassVar
from utils.base_discretizer import BaseDiscretizer

from utils.config import (
    ACROBOT_DEFAULT_BINS,
    ACROBOT_DEFAULT_HIGH,
    ACROBOT_DEFAULT_LOW,
)

@dataclass(slots=True)
class AcrobotDiscretizer(BaseDiscretizer):
    """Converte un'osservazione continua di Acrobot in un indice discreto.

    L'osservazione di Acrobot-v1 e' un vettore a sei dimensioni:
    [cos(theta1), sin(theta1), cos(theta2), sin(theta2),
     velocita' angolare di theta1, velocita' angolare di theta2].
    """

    bins_per_dimension: Sequence[int] = ACROBOT_DEFAULT_BINS
    low: Sequence[float] = ACROBOT_DEFAULT_LOW
    high: Sequence[float] = ACROBOT_DEFAULT_HIGH

    expected_dim: ClassVar[int] = 6