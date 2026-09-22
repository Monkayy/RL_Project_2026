"""Analisi di sensibilita' degli iperparametri per Acrobot-v1."""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from acrobot.discretizer import AcrobotDiscretizer
from acrobot.policy import make_greedy_policy
from algorithms.monte_carlo import monte_carlo_prediction
from algorithms.q_learning import QLearningConfig, q_learning
from algorithms.sarsa import SARSAConfig, sarsa
from algorithms.td_zero import td_zero_prediction
from utils.config import (
    ACROBOT_SENSITIVITY_EPISODES,
    ACROBOT_DEFAULT_BINS,
    ACROBOT_DEFAULT_HIGH,
    ACROBOT_DEFAULT_LOW,
    ACROBOT_MONTE_CARLO_PARAMS,
    ACROBOT_Q_LEARNING_PARAMS,
    ACROBOT_SARSA_PARAMS,
    ACROBOT_TD_ZERO_PARAMS,
    DATA_DIR,
    ENV_ID_ACROBOT,
    FIGURE_DIR,
    SEEDS,
)

PARAMETERS = {
    "alpha": (0.05, 0.10, 0.20, 0.50),
    "gamma": (0.80, 0.90, 0.99, 1.00),
    "epsilon": (0.01, 0.05, 0.10, 0.30),
}


def make_discretizer():
    return AcrobotDiscretizer(
        bins_per_dimension=ACROBOT_DEFAULT_BINS,
        low=ACROBOT_DEFAULT_LOW,
        high=ACROBOT_DEFAULT_HIGH,
    )


def run_control_parameter(name, parameter, values, episodes, seeds):
    print(f"\n" + "=" * 60)
    print(f" START CONTROL STUDY: {name.upper()} | Parameter: {parameter}")
    print(f"  Values to test: {values}")
    print(f"  Episodes: {episodes} | Seeds ({len(seeds)}): {seeds}")
    print("=" * 60)

    base = dict(
        ACROBOT_Q_LEARNING_PARAMS
        if name == "q_learning"
        else ACROBOT_SARSA_PARAMS
    )
    curves = np.zeros((len(values), len(seeds), episodes))
    algorithm = q_learning if name == "q_learning" else sarsa
    config_type = QLearningConfig if name == "q_learning" else SARSAConfig
    study_start_time = time.time()

    for value_index, value in enumerate(values):
        params = dict(base)
        if parameter == "epsilon":
            params.update(
                epsilon_start=value, epsilon_min=value, epsilon_decay=1.0
            )
        else:
            params[parameter] = value

        print(f"\n  [{value_index + 1}/{len(values)}] Running {name} with {parameter}={value}...")
        step_start_time = time.time()

        for seed_index, seed in enumerate(seeds):
            config = config_type(num_episodes=episodes, **params)
            result = algorithm(ENV_ID_ACROBOT, make_discretizer(), seed, config)
            curves[value_index, seed_index] = result.episode_returns

        elapsed = time.time() - step_start_time
        mean_ret = curves[value_index].mean(axis=0)[-min(100, episodes):].mean()
        print(f"    Done in {elapsed:.2f}s across {len(seeds)} seeds | Final Mean Return: {mean_ret:.2f}")

    total_time = time.time() - study_start_time
    print(f"\n FINISHED CONTROL STUDY: {name} ({parameter}) in {total_time:.2f}s")
    return curves


def run_prediction_parameter(algorithm, parameter, values, episodes, seeds):
    print(f"\n" + "=" * 60)
    print(f" START PREDICTION STUDY: {algorithm.upper()} | Parameter: {parameter}")
    print(f"  Values to test: {values}")
    print(f"  Episodes: {episodes} | Seeds ({len(seeds)}): {seeds}")
    print("=" * 60)

    curves = np.zeros((len(values), len(seeds), episodes))
    study_start_time = time.time()

    for value_index, value in enumerate(values):
        print(f"\n  [{value_index + 1}/{len(values)}] Running {algorithm} with {parameter}={value}...")
        step_start_time = time.time()

        for seed_index, seed in enumerate(seeds):
            discretizer = make_discretizer()
            baseline = q_learning(
                ENV_ID_ACROBOT,
                discretizer,
                seed,
                QLearningConfig(num_episodes=episodes, **ACROBOT_Q_LEARNING_PARAMS),
            )
            policy = make_greedy_policy(
                baseline.q_table, discretizer, np.random.default_rng(seed)
            )
            params = dict(
                ACROBOT_MONTE_CARLO_PARAMS
                if algorithm == "mc"
                else ACROBOT_TD_ZERO_PARAMS
            )
            params[parameter] = value

            if algorithm == "mc":
                result = monte_carlo_prediction(
                    ENV_ID_ACROBOT, discretizer, policy, episodes, seed, **params
                )
            else:
                result = td_zero_prediction(
                    ENV_ID_ACROBOT, discretizer, policy, episodes, seed, **params
                )
            curves[value_index, seed_index] = result.mean_absolute_updates

        elapsed = time.time() - step_start_time
        mean_upd = curves[value_index].mean(axis=0)[-min(100, episodes):].mean()
        print(f"    Done in {elapsed:.2f}s across {len(seeds)} seeds | Final Mean Absolute Update: {mean_upd:.4f}")

    total_time = time.time() - study_start_time
    print(f"\n FINISHED PREDICTION STUDY: {algorithm} ({parameter}) in {total_time:.2f}s")
    return curves


def save_and_plot(name, parameter, values, curves, ylabel, data_path=None):
    if data_path is None:
        data_path = DATA_DIR / f"acrobot_{name}_sensitivity_{parameter}.npz"
    figure_path = FIGURE_DIR / f"acrobot_{name}_sensitivity_{parameter}.png"
    if data_path.parent == DATA_DIR:
        np.savez(data_path, values=np.asarray(values), curves=curves)

    mean = curves.mean(axis=1)
    std = curves.std(axis=1)
    episodes = np.arange(1, curves.shape[-1] + 1)

    final_window = min(100, curves.shape[-1])
    final_by_seed = curves[:, :, -final_window:].mean(axis=2)
    final_mean = final_by_seed.mean(axis=1)
    final_std = final_by_seed.std(axis=1)
    higher_is_better = name not in {"mc", "td0"}
    best_index = int(
        np.argmax(final_mean) if higher_is_better else np.argmin(final_mean)
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        gridspec_kw={"width_ratios": (1.55, 1)},
    )

    learning_axis, sensitivity_axis = axes
    for index, value in enumerate(values):
        learning_axis.plot(
            episodes,
            mean[index],
            label=f"{parameter}={value:g}",
            linewidth=1.8,
        )
        learning_axis.fill_between(
            episodes, mean[index] - std[index], mean[index] + std[index], alpha=0.12
        )

    learning_axis.set_xlabel("Episodio")
    learning_axis.set_ylabel(ylabel)
    learning_axis.set_title("Andamento durante il training")
    learning_axis.grid(alpha=0.2)
    learning_axis.legend(title=parameter, ncol=2, fontsize=9)

    sensitivity_axis.errorbar(
        values,
        final_mean,
        yerr=final_std,
        fmt="o-",
        color="#2F6690",
        ecolor="#7A9EAF",
        elinewidth=1.5,
        capsize=4,
        markersize=6,
        linewidth=2,
    )
    sensitivity_axis.scatter(
        [values[best_index]],
        [final_mean[best_index]],
        s=90,
        color="#D1495B",
        zorder=3,
        label=f"Migliore: {values[best_index]:g}",
    )
    sensitivity_axis.set_xticks(values)
    sensitivity_axis.set_xlabel(parameter)
    sensitivity_axis.set_ylabel(
        f"{ylabel} medio (ultimi {final_window} episodi)"
    )
    sensitivity_axis.set_title("Sensibilita' del risultato finale")
    sensitivity_axis.grid(alpha=0.2)
    sensitivity_axis.legend(fontsize=9)

    figure.suptitle(f"Acrobot-v1: {name} - sensibilita' a {parameter}")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close()

    print(f"\n Saved Figure: {figure_path}")
    print(f" Source Data : {data_path}")
    print(" Summary Scores (Last 100 episodes):")
    for value, score, spread in zip(values, final_mean, final_std):
        print(
            f"     - {name} {parameter}={value}: "
            f"{score:.4f} +/- {spread:.4f}"
        )


def plot_saved_results():
    """Rigenera i grafici dagli NPZ senza rieseguire alcun esperimento."""
    search_directories = (
        DATA_DIR,
        DATA_DIR.parent.parent / "utils" / "outputs" / "data",
    )
    files = sorted(
        {
            path
            for directory in search_directories
            if directory.exists()
            for path in directory.glob("acrobot_*_sensitivity_*.npz")
        }
    )
    if not files:
        raise FileNotFoundError(
            "Nessun archivio di sensibilita' trovato in: "
            + ", ".join(str(directory) for directory in search_directories)
        )

    for path in files:
        stem = path.stem.removeprefix("acrobot_")
        name, parameter = stem.split("_sensitivity_", maxsplit=1)
        data = np.load(path)
        save_and_plot(
            name=name,
            parameter=parameter,
            values=data["values"],
            curves=data["curves"],
            ylabel=(
                "Aggiornamento medio assoluto"
                if name in {"mc", "td0"}
                else "Return"
            ),
            data_path=path,
        )

    print(f"\nRigenerati {len(files)} grafici senza nuovo training.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="rigenera i grafici dagli NPZ senza rieseguire gli esperimenti",
    )
    parser.add_argument(
        "--episodes", type=int, default=ACROBOT_SENSITIVITY_EPISODES
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    if args.plot_only:
        plot_saved_results()
        return
    if args.episodes < 1:
        raise ValueError("--episodes deve essere positivo")

    global_start_time = time.time()

    print("\n" + "#" * 60)
    print(" ACROBOT-V1 HYPERPARAMETER SENSITIVITY ANALYSIS ")
    print("#" * 60)
    print(f"Environment ID : {ENV_ID_ACROBOT}")
    print(f"Episodes/Run   : {args.episodes}")
    print(f"Seeds Count    : {len(args.seeds)} ({args.seeds})")
    print(f"Output Path    : {FIGURE_DIR}")

    for name in ("q_learning", "sarsa"):
        for parameter, values in PARAMETERS.items():
            curves = run_control_parameter(
                name, parameter, values, args.episodes, args.seeds
            )
            save_and_plot(name, parameter, values, curves, "Return")

    for name in ("mc", "td0"):
        for parameter in ("alpha", "gamma"):
            values = PARAMETERS[parameter]
            curves = run_prediction_parameter(
                name, parameter, values, args.episodes, args.seeds
            )
            save_and_plot(
                name, parameter, values, curves, "Aggiornamento medio assoluto"
            )

    total_time = time.time() - global_start_time
    print("\n" + "#" * 60)
    print(f" ALL ACROBOT SENSITIVITY EXPERIMENTS COMPLETED IN {total_time / 60:.2f} MINUTES ")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()