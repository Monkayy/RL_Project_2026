"""Grafici per gli esperimenti su Acrobot.

Legge i file acrobot_seed_<seed>.npz salvati da run_acrobot_experiment.py
in config.DATA_DIR e produce i grafici richiesti dal progetto (learning
curves, confronto fra algoritmi, stabilita') piu' alcuni diagnostici
specifici per Acrobot (tasso di troncamento, decadimento di epsilon).

NON prodotti qui (richiedono run aggiuntive o instrumentazione non
ancora presente):
- Sensitivity analysis rispetto ad alpha/gamma/epsilon_decay: va fatta
  eseguendo run_acrobot_experiment con diverse combinazioni di
  ACROBOT_*_PARAMS e confrontando i risultati salvati.
- Copertura dello spazio degli stati NEL TEMPO: le tabelle attuali
  salvano solo lo stato finale delle q_table/v_table, non quali stati
  sono stati visitati episodio per episodio. Qui viene mostrata solo
  la copertura finale (stati con valore non nullo a fine training).

Uso:
    python -m acrobot.plot_acrobot_results
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.config import ACROBOT_DEFAULT_BINS, DATA_DIR, FIGURE_DIR, SEEDS

MAX_EPISODE_STEPS = 500
ROLLING_WINDOW = 100


def rolling_mean(x, window):
    window = max(1, min(window, len(x)))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def load_all(seeds):
    data = {}
    for seed in seeds:
        path = DATA_DIR / f"acrobot_seed_{seed}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} non trovato: esegui prima "
                f"run_acrobot_experiment.py."
            )
        data[seed] = np.load(path)
    return data


def stack_smoothed(data, key, window=ROLLING_WINDOW):
    curves = [rolling_mean(d[key], window) for d in data.values()]
    min_len = min(len(c) for c in curves)
    return np.stack([c[:min_len] for c in curves])


def save_fig(name):
    out_path = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Salvato: {out_path}")


# --------------------------------------------------------------------
# 1. Learning curve singola (media +/- std sui seed)
# --------------------------------------------------------------------

def plot_learning_curve(data, key, label, out_name):
    curves = stack_smoothed(data, key)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0)
    x = np.arange(len(mean))

    plt.figure(figsize=(8, 5))
    plt.plot(x, mean, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel(f"Episodio (media mobile, finestra={ROLLING_WINDOW})")
    plt.ylabel("Return")
    plt.title(f"{label} - Acrobot-v1 (media +/- std su {len(data)} seed)")
    plt.legend()
    save_fig(out_name)


# --------------------------------------------------------------------
# 2. Confronto fra algoritmi (overlay, media +/- std sui seed)
# --------------------------------------------------------------------

def plot_comparison(data, keys_and_labels, ylabel, title, out_name):
    plt.figure(figsize=(8, 5))
    for key, label in keys_and_labels:
        curves = stack_smoothed(data, key)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)

    plt.xlabel(f"Episodio (media mobile, finestra={ROLLING_WINDOW})")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    save_fig(out_name)


# --------------------------------------------------------------------
# 3. Tasso di troncamento (diagnostico difficolta' di esplorazione)
# --------------------------------------------------------------------

def plot_truncation_rate(data, key_lengths, label, out_name):
    plt.figure(figsize=(8, 5))
    for seed, d in data.items():
        truncated = (d[key_lengths] >= MAX_EPISODE_STEPS).astype(np.float64)
        rate = rolling_mean(truncated, ROLLING_WINDOW)
        plt.plot(rate, label=f"seed {seed}", alpha=0.8)

    plt.xlabel(f"Episodio (media mobile, finestra={ROLLING_WINDOW})")
    plt.ylabel("Frazione di episodi troncati (500 step)")
    plt.title(f"{label} - tasso di troncamento per episodio")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    save_fig(out_name)


# --------------------------------------------------------------------
# 4. Decadimento di epsilon (verifica visiva)
# --------------------------------------------------------------------

def plot_epsilon_decay(data, key_epsilons, label, out_name):
    plt.figure(figsize=(8, 5))
    for seed, d in data.items():
        plt.plot(d[key_epsilons], label=f"seed {seed}")

    plt.xlabel("Episodio")
    plt.ylabel("epsilon")
    plt.title(f"{label} - decadimento di epsilon")
    plt.legend()
    save_fig(out_name)


# --------------------------------------------------------------------
# 5. Convergenza (mean_absolute_updates) - MC vs TD(0)
# --------------------------------------------------------------------

def plot_convergence_comparison(data, out_name):
    plot_comparison(
        data,
        [("mc_updates", "Monte Carlo"), ("td0_updates", "TD(0)")],
        ylabel="Aggiornamento medio assoluto |delta V|",
        title="Convergenza: Monte Carlo vs TD(0) - Acrobot-v1",
        out_name=out_name,
    )


# --------------------------------------------------------------------
# 6. Distribuzione di V(s) - MC vs TD(0), stati visitati
# --------------------------------------------------------------------

def plot_v_distribution(data, out_name):
    mc_values = np.concatenate(
        [d["mc_v_table"][d["mc_v_table"] != 0].ravel() for d in data.values()]
    )
    td0_values = np.concatenate(
        [d["td0_v_table"][d["td0_v_table"] != 0].ravel() for d in data.values()]
    )

    plt.figure(figsize=(8, 5))
    plt.hist(mc_values, bins=40, alpha=0.5, label="Monte Carlo", density=True)
    plt.hist(td0_values, bins=40, alpha=0.5, label="TD(0)", density=True)
    plt.xlabel("V(s)")
    plt.ylabel("Densita'")
    plt.title(
        "Distribuzione di V(s) sugli stati visitati - "
        "Monte Carlo vs TD(0) (tutti i seed)"
    )
    plt.legend()
    save_fig(out_name)

    print(
        f"MC:    media={mc_values.mean():8.2f}  std={mc_values.std():7.2f}  "
        f"n_stati={mc_values.size}"
    )
    print(
        f"TD(0): media={td0_values.mean():8.2f}  std={td0_values.std():7.2f}  "
        f"n_stati={td0_values.size}"
    )


# --------------------------------------------------------------------
# 7. Copertura finale dello spazio degli stati
# --------------------------------------------------------------------

def print_state_coverage(data):
    total_states = int(np.prod(ACROBOT_DEFAULT_BINS))

    print(f"\nSpazio degli stati discretizzato: {total_states} stati totali.")
    for seed, d in data.items():
        ql_visited = int(np.count_nonzero(np.any(d["ql_q_table"] != 0, axis=-1)))
        mc_visited = int(np.count_nonzero(d["mc_v_table"] != 0))
        td0_visited = int(np.count_nonzero(d["td0_v_table"] != 0))
        print(
            f"seed {seed}: Q-Learning {ql_visited}/{total_states} "
            f"({100 * ql_visited / total_states:5.1f}%)  |  "
            f"MC {mc_visited}/{total_states} "
            f"({100 * mc_visited / total_states:5.1f}%)  |  "
            f"TD(0) {td0_visited}/{total_states} "
            f"({100 * td0_visited / total_states:5.1f}%)"
        )


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

def main():
    data = load_all(SEEDS)

    plot_learning_curve(
        data, "ql_returns", "Q-Learning", "acrobot_q_learning_curve.png"
    )
    plot_learning_curve(data, "sarsa_returns", "SARSA", "acrobot_sarsa_curve.png")

    plot_comparison(
        data,
        [("ql_returns", "Q-Learning"), ("sarsa_returns", "SARSA")],
        ylabel="Return",
        title="Q-Learning vs SARSA - Acrobot-v1",
        out_name="acrobot_qlearning_vs_sarsa.png",
    )

    plot_truncation_rate(
        data, "ql_lengths", "Q-Learning", "acrobot_ql_truncation_rate.png"
    )
    plot_truncation_rate(
        data, "sarsa_lengths", "SARSA", "acrobot_sarsa_truncation_rate.png"
    )

    plot_epsilon_decay(
        data, "ql_epsilons", "Q-Learning", "acrobot_ql_epsilon.png"
    )

    plot_convergence_comparison(data, "acrobot_mc_vs_td0_convergence.png")
    plot_v_distribution(data, "acrobot_v_distribution.png")

    print_state_coverage(data)

    print("\nTutti i grafici sono stati salvati in", FIGURE_DIR)


if __name__ == "__main__":
    main()