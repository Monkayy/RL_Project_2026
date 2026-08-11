"""Test rapido dei quattro algoritmi su CartPole."""

from utils.config import (
    DEFAULT_BINS,
    DEFAULT_HIGH,
    DEFAULT_LOW,
    ENV_ID,
)

from cartpole.discretizer import CartPoleDiscretizer

from algorithms.q_learning import q_learning, QLearningConfig
from algorithms.sarsa import sarsa

from algorithms.monte_carlo import (
    monte_carlo_prediction,
)

from algorithms.td_zero import (
    td_zero_prediction,
)


TEST_EPISODES = 20
TEST_SEED = 42


def main() -> None:
    print("Creazione del discretizzatore...")

    discretizer = CartPoleDiscretizer(
        bins_per_dimension=DEFAULT_BINS,
        low=DEFAULT_LOW,
        high=DEFAULT_HIGH,
    )

    print(
        "Forma dello spazio discreto:",
        discretizer.state_shape,
    )

    print("\n1. Test Q-Learning")


    q_result = q_learning(
        env_id=ENV_ID,
        discretizer=discretizer,
        seed=TEST_SEED,
        config=QLearningConfig(num_episodes=TEST_EPISODES)
    )

    print(
        "Forma Q-table:",
        q_result.q_table.shape,
    )

    print(
        "Reward Q-Learning:",
        q_result.episode_returns,
    )

    print("\n2. Test SARSA")

    sarsa_result = sarsa(
        env_id=ENV_ID,
        discretizer=discretizer,
        num_episodes=TEST_EPISODES,
        seed=TEST_SEED,
        alpha=0.2,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.95,
    )

    print(
        "Forma Q-table:",
        sarsa_result.q_table.shape,
    )

    print(
        "Reward SARSA:",
        sarsa_result.episode_returns,
    )

    print("\n3. Test Monte Carlo Prediction")

    mc_result = monte_carlo_prediction(
        env_id=ENV_ID,
        discretizer=discretizer,
        num_episodes=TEST_EPISODES,
        seed=TEST_SEED,
        alpha=0.05,
        gamma=0.99,
        first_visit=True,
    )

    print(
        "Forma V-table:",
        mc_result.v_table.shape,
    )

    print(
        "Reward Monte Carlo:",
        mc_result.episode_returns,
    )

    print("\n4. Test TD(0) Prediction")

    td_result = td_zero_prediction(
        env_id=ENV_ID,
        discretizer=discretizer,
        num_episodes=TEST_EPISODES,
        seed=TEST_SEED,
        alpha=0.05,
        gamma=0.99,
    )

    print(
        "Forma V-table:",
        td_result.v_table.shape,
    )

    print(
        "Reward TD(0):",
        td_result.episode_returns,
    )

    print("\nTutti i test sono stati completati correttamente.")


if __name__ == "__main__":
    main()