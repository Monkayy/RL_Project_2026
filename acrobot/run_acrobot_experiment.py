"""Script di verifica per gli esperimenti su Acrobot.

Utilizzo:
    python run_acrobot_experiment.py
    python run_acrobot_experiment.py --episodes 200 --seeds 42   # test veloce
"""

import argparse
import sys
import time

import gymnasium as gym
import numpy as np

from acrobot.discretizer import AcrobotDiscretizer
from acrobot.policy import make_greedy_policy
from algorithms.q_learning import q_learning
from algorithms.sarsa import sarsa
from algorithms.monte_carlo import monte_carlo_prediction
from algorithms.td_zero import td_zero_prediction

from utils.config import (
    ACROBOT_CONTROL_EPISODES,
    ACROBOT_DEFAULT_BINS,
    ACROBOT_DEFAULT_HIGH,
    ACROBOT_DEFAULT_LOW,
    ACROBOT_PREDICTION_EPISODES,
    ACROBOT_Q_LEARNING_PARAMS,
    ACROBOT_SARSA_PARAMS,
    DATA_DIR,
    ENV_ID_ACROBOT,
    SEEDS,
)

MAX_EPISODE_STEPS = 500

class VerificationError(AssertionError):
    """Sollevata quando un controllo di correttezza fallisce."""

# --------------------------------------------------------------------
# Controlli di correttezza
# --------------------------------------------------------------------

def check_control_result(name, result, discretizer, n_actions, epsilon_min):
    problems = []

    if not np.isfinite(result.q_table).all():
        problems.append("q_table contiene NaN o Inf.")

    expected_shape = (*discretizer.state_shape, n_actions)
    if result.q_table.shape != expected_shape:
        problems.append(
            f"q_table ha shape {result.q_table.shape}, "
            f"attesa {expected_shape}."
        )

    if np.any(result.episode_lengths < 1) or np.any(
        result.episode_lengths > MAX_EPISODE_STEPS
    ):
        problems.append(
            f"episode_lengths fuori dal range [1, {MAX_EPISODE_STEPS}]."
        )

    if np.any(result.episode_returns > 0):
        problems.append(
            "episode_returns positivo: in Acrobot il reward per "
            "step e' -1 (0 solo alla transizione di successo), "
            "quindi il ritorno non puo' mai essere positivo."
        )

    if result.epsilons[-1] > epsilon_min + 1e-9:
        problems.append(
            f"epsilon finale ({result.epsilons[-1]:.4f}) non ha "
            f"raggiunto epsilon_min ({epsilon_min}): aumenta "
            f"num_episodes o riduci epsilon_decay."
        )

    if problems:
        raise VerificationError(
            f"[{name}] controlli falliti:\n  - " + "\n  - ".join(problems)
        )


def check_prediction_result(name, result, discretizer):
    problems = []

    if not np.isfinite(result.v_table).all():
        problems.append("v_table contiene NaN o Inf.")

    if result.v_table.shape != discretizer.state_shape:
        problems.append(
            f"v_table ha shape {result.v_table.shape}, "
            f"attesa {discretizer.state_shape}."
        )

    if np.any(result.episode_lengths < 1) or np.any(
        result.episode_lengths > MAX_EPISODE_STEPS
    ):
        problems.append(
            f"episode_lengths fuori dal range [1, {MAX_EPISODE_STEPS}]."
        )

    if np.any(result.episode_returns > 0):
        problems.append(
            "episode_returns positivo (non atteso per Acrobot)."
        )

    if problems:
        raise VerificationError(
            f"[{name}] controlli falliti:\n  - " + "\n  - ".join(problems)
        )


def get_n_actions(env_id):
    env = gym.make(env_id)
    n_actions = int(env.action_space.n)
    env.close()
    return n_actions


# --------------------------------------------------------------------
# Esecuzione per un singolo seed
# --------------------------------------------------------------------

def run_single_seed(seed, n_actions, control_episodes, prediction_episodes):
    print(f"\n=== Seed {seed} ===")

    discretizer = AcrobotDiscretizer(
        bins_per_dimension=ACROBOT_DEFAULT_BINS,
        low=ACROBOT_DEFAULT_LOW,
        high=ACROBOT_DEFAULT_HIGH,
    )

    t0 = time.time()
    ql_result = q_learning(
        ENV_ID_ACROBOT, discretizer, control_episodes, seed
    )
    check_control_result(
        "Q-Learning",
        ql_result,
        discretizer,
        n_actions=n_actions,
        epsilon_min=ACROBOT_Q_LEARNING_PARAMS["epsilon_min"],
    )
    print(
        f"Q-Learning        ok  ({time.time() - t0:6.1f}s)  "
        f"return medio (ultimi 100 ep): "
        f"{ql_result.episode_returns[-100:].mean():8.2f}"
    )

    t0 = time.time()
    sarsa_result = sarsa(
        ENV_ID_ACROBOT, discretizer, control_episodes, seed
    )
    check_control_result(
        "SARSA",
        sarsa_result,
        discretizer,
        n_actions=n_actions,
        epsilon_min=ACROBOT_SARSA_PARAMS["epsilon_min"],
    )
    print(
        f"SARSA             ok  ({time.time() - t0:6.1f}s)  "
        f"return medio (ultimi 100 ep): "
        f"{sarsa_result.episode_returns[-100:].mean():8.2f}"
    )

    # La policy valutata da MC/TD(0) e' quella greedy rispetto alla
    # Q-table di Q-Learning (vedi acrobot/policies.py).
    rng = np.random.default_rng(seed)

    greedy_policy = make_greedy_policy(ql_result.q_table, discretizer, rng)

    t0 = time.time()
    mc_result = monte_carlo_prediction(
        ENV_ID_ACROBOT,
        discretizer,
        greedy_policy,
        prediction_episodes,
        seed,
    )
    check_prediction_result("MC Prediction", mc_result, discretizer)
    visited_mc = mc_result.v_table[mc_result.v_table != 0]
    print(
        f"MC Prediction     ok  ({time.time() - t0:6.1f}s)  "
        f"V medio (stati visitati): "
        f"{visited_mc.mean() if visited_mc.size else float('nan'):8.2f}"
    )

    t0 = time.time()
    td0_result = td_zero_prediction(
        ENV_ID_ACROBOT,
        discretizer,
        greedy_policy,
        prediction_episodes,
        seed,
    )
    check_prediction_result("TD(0) Prediction", td0_result, discretizer)
    visited_td0 = td0_result.v_table[td0_result.v_table != 0]
    print(
        f"TD(0) Prediction  ok  ({time.time() - t0:6.1f}s)  "
        f"V medio (stati visitati): "
        f"{visited_td0.mean() if visited_td0.size else float('nan'):8.2f}"
    )

    return {
        "q_learning": ql_result,
        "sarsa": sarsa_result,
        "mc_prediction": mc_result,
        "td_zero_prediction": td0_result,
    }


# --------------------------------------------------------------------
# Aggregazione fra seed e salvataggio
# --------------------------------------------------------------------

def summarize_returns(all_results, algorithm):
    finals = [
        r[algorithm].episode_returns[-100:].mean() for r in all_results
    ]
    return float(np.mean(finals)), float(np.std(finals))


def save_results(seeds, all_results):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for seed, results in zip(seeds, all_results):
        out_path = DATA_DIR / f"acrobot_seed_{seed}.npz"
        np.savez(
            out_path,
            ql_q_table=results["q_learning"].q_table,
            ql_returns=results["q_learning"].episode_returns,
            ql_lengths=results["q_learning"].episode_lengths,
            ql_epsilons=results["q_learning"].epsilons,
            sarsa_q_table=results["sarsa"].q_table,
            sarsa_returns=results["sarsa"].episode_returns,
            sarsa_lengths=results["sarsa"].episode_lengths,
            sarsa_epsilons=results["sarsa"].epsilons,
            mc_v_table=results["mc_prediction"].v_table,
            mc_returns=results["mc_prediction"].episode_returns,
            mc_lengths=results["mc_prediction"].episode_lengths,
            mc_updates=results["mc_prediction"].mean_absolute_updates,
            td0_v_table=results["td_zero_prediction"].v_table,
            td0_returns=results["td_zero_prediction"].episode_returns,
            td0_lengths=results["td_zero_prediction"].episode_lengths,
            td0_updates=results["td_zero_prediction"].mean_absolute_updates,
        )
        print(f"Salvato: {out_path}")


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Esegue e verifica la pipeline RL completa su Acrobot."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "Sovrascrive il numero di episodi per tutti gli algoritmi "
            "(utile per un test veloce, es. --episodes 200)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Sovrascrive i seed da usare (default: config.SEEDS).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    seeds = tuple(args.seeds) if args.seeds else SEEDS
    control_episodes = args.episodes or ACROBOT_CONTROL_EPISODES
    prediction_episodes = args.episodes or ACROBOT_PREDICTION_EPISODES

    print(
        f"Verifica pipeline Acrobot su {len(seeds)} seed: {seeds}\n"
        f"Episodi controllo: {control_episodes}, "
        f"episodi prediction: {prediction_episodes}"
    )

    n_actions = get_n_actions(ENV_ID_ACROBOT)

    all_results = []
    for seed in seeds:
        try:
            all_results.append(
                run_single_seed(
                    seed, n_actions, control_episodes, prediction_episodes
                )
            )
        except VerificationError as exc:
            print(f"\nFALLITO per seed {seed}:\n{exc}")
            sys.exit(1)

    print("\n=== Riepilogo (media +/- std sui seed, ultimi 100 episodi) ===")
    for algorithm in (
        "q_learning",
        "sarsa",
        "mc_prediction",
        "td_zero_prediction",
    ):
        mean, std = summarize_returns(all_results, algorithm)
        print(f"{algorithm:20s}: {mean:8.2f} +/- {std:6.2f}")

    save_results(seeds, all_results)

    print("\nTutti i controlli sono passati.")


if __name__ == "__main__":
    main()