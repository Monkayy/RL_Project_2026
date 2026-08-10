"""Strutture dati restituite dagli algoritmi."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ControlResult:
    q_table: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    epsilons: np.ndarray


@dataclass(slots=True)
class PredictionResult:
    v_table: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    mean_absolute_updates: np.ndarray