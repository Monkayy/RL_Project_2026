"""Configurazioni condivise per gli esperimenti DQN su Breakout."""

from dataclasses import dataclass
from pathlib import Path

ENV_ID = "ALE/Breakout-v5"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "breakout"
DATA_DIR = OUTPUT_DIR / "data"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

for directory in (DATA_DIR, FIGURE_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DQNConfig:
    total_steps: int = 250_000
    replay_capacity: int = 50_000
    replay_start_size: int = 10_000
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 1e-4
    train_frequency: int = 4
    target_update_frequency: int = 5_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    evaluation_frequency: int = 25_000
    evaluation_episodes: int = 10
    max_grad_norm: float = 10.0
    use_replay: bool = True
    use_target_network: bool = True

    def epsilon(self, step: int) -> float:
        fraction = min(step / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
