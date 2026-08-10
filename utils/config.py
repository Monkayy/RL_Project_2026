"""Configurazione centrale del Compito 1: CartPole."""

from pathlib import Path

ENV_ID = "CartPole-v1"
SEEDS = (42, 123, 456)

# Numero di episodi consigliato per gli esperimenti completi.
CONTROL_EPISODES = 3000
PREDICTION_EPISODES = 3000
SENSITIVITY_EPISODES = 2000

# Discretizzazione di:
# [posizione, velocità, angolo, velocità angolare].
DEFAULT_BINS = (8, 8, 12, 12)

# Limiti utilizzati dalla discretizzazione.
# Le velocità hanno teoricamente range infinito, quindi scegliamo
# intervalli finiti ragionevoli e tagliamo i valori esterni.
DEFAULT_LOW = (-2.4, -3.0, -0.2095, -4.0)
DEFAULT_HIGH = (2.4, 3.0, 0.2095, 4.0)

Q_LEARNING_PARAMS = {
    "alpha": 0.20,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.997,
}

SARSA_PARAMS = {
    "alpha": 0.20,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.997,
}

MONTE_CARLO_PARAMS = {
    "alpha": 0.05,
    "gamma": 0.99,
    "first_visit": True,
}

TD_ZERO_PARAMS = {
    "alpha": 0.05,
    "gamma": 0.99,
}

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"

for directory in (OUTPUT_DIR, FIGURE_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)




######### ACROBOT CONFIG #########

ENV_ID_ACROBOT: str = "Acrobot-v1"

ACROBOT_CONTROL_EPISODES: int = 8000
ACROBOT_PREDICTION_EPISODES: int = 8000
ACROBOT_SENSITIVITY_EPISODES: int = 4000

# Discretizzazione di:
# [cos(theta1), sin(theta1), cos(theta2), sin(theta2),
#  velocita' angolare theta1, velocita' angolare theta2].
ACROBOT_DEFAULT_BINS = (6, 6, 6, 6, 7, 7)

# Limiti dello spazio delle osservazioni
ACROBOT_DEFAULT_LOW = (
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -12.566371,
    -28.274334,
)
ACROBOT_DEFAULT_HIGH = (
    1.0,
    1.0,
    1.0,
    1.0,
    12.566371,
    28.274334,
)

ACROBOT_Q_LEARNING_PARAMS = {
    "alpha": 0.20,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.999,
}

ACROBOT_SARSA_PARAMS = {
    "alpha": 0.20,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.999,
}

ACROBOT_MONTE_CARLO_PARAMS = {
    "alpha": 0.05,
    "gamma": 0.99,
    "first_visit": True,
}

ACROBOT_TD_ZERO_PARAMS = {
    "alpha": 0.05,
    "gamma": 0.99,
}