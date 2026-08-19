"""Grafici per gli esperimenti su Acrobot.

Legge i file acrobot_seed_<seed>.npz salvati da run_acrobot_experiment.py
in config.DATA_DIR e produce i grafici richiesti dal progetto (learning
curves, confronto fra algoritmi, stabilita') piu' alcuni diagnostici
specifici per Acrobot (tasso di troncamento, decadimento di epsilon).

La sensitivity analysis viene prodotta da acrobot.sensitivity e salva
un file .npz e un grafico per ogni algoritmo/parametro.
- Copertura dello spazio degli stati NEL TEMPO: le tabelle attuali
  salvano solo lo stato finale delle q_table/v_table, non quali stati
  sono stati visitati episodio per episodio. Qui viene mostrata solo
  la copertura finale (stati con valore non nullo a fine training).

Uso:
    python -m acrobot.plot_acrobot_results
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from utils.config import ACROBOT_DEFAULT_BINS, DATA_DIR, FIGURE_DIR, SEEDS

MAX_EPISODE_STEPS = 500
ROLLING_WINDOW = 100


def rolling_mean(x, window):
    window = max(1, min(window, len(x)))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


SERIES_KEYS = (
    "ql_returns",
    "sarsa_returns",
    "mc_returns",
    "td0_returns",
    "ql_lengths",
    "sarsa_lengths",
    "mc_lengths",
    "td0_lengths",
    "mc_updates",
    "td0_updates",
)


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


def validate_data(data):
    """Evita confronti tra run brevi e run completi."""
    reference_length = None
    for seed, result in data.items():
        lengths = {key: len(result[key]) for key in SERIES_KEYS}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(
                f"Il file del seed {seed} contiene serie di lunghezza "
                f"incoerente: {lengths}. Riesegui quel seed."
            )

        length = unique_lengths.pop()
        if reference_length is None:
            reference_length = length
        elif length != reference_length:
            raise ValueError(
                "I seed hanno un numero di episodi differente "
                f"({reference_length} e {length}). Non generare grafici "
                "aggregati: riesegui il/i seed incompleti con lo stesso "
                "valore di --episodes."
            )

    return reference_length


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


def plot_final_control_performance(data, out_name, window=ROLLING_WINDOW):
    """Confronta la performance finale sui seed, non solo le curve."""
    labels = ("Q-Learning", "SARSA")
    keys = ("ql_returns", "sarsa_returns")
    scores = np.asarray(
        [[result[key][-window:].mean() for result in data.values()] for key in keys]
    )

    plt.figure(figsize=(7, 5))
    positions = np.arange(len(labels))
    plt.bar(
        positions,
        scores.mean(axis=1),
        yerr=scores.std(axis=1),
        capsize=5,
        color=("#4C78A8", "#F58518"),
    )
    plt.xticks(positions, labels)
    plt.ylabel(f"Return medio (ultimi {window} episodi)")
    plt.title("Performance finale: Q-Learning vs SARSA - Acrobot-v1")
    save_fig(out_name)


def write_summary(data, window=ROLLING_WINDOW):
    """Salva numeri citabili nella relazione insieme ai grafici."""
    out_path = DATA_DIR / "acrobot_summary.txt"
    lines = [
        "Acrobot-v1 — riepilogo sperimentale",
        f"Seed: {', '.join(str(seed) for seed in data)}",
        f"Finestra finale: {window} episodi",
        "",
        "Algoritmo | return iniziale | return finale | std finale | troncati finali",
        "-" * 78,
    ]
    for label, returns_key, lengths_key in (
        ("Q-Learning", "ql_returns", "ql_lengths"),
        ("SARSA", "sarsa_returns", "sarsa_lengths"),
        ("Monte Carlo prediction", "mc_returns", "mc_lengths"),
        ("TD(0) prediction", "td0_returns", "td0_lengths"),
    ):
        returns = np.stack([result[returns_key] for result in data.values()])
        lengths = np.stack([result[lengths_key] for result in data.values()])
        initial = returns[:, :window].mean()
        final = returns[:, -window:].mean()
        final_std = returns[:, -window:].std()
        truncated = (lengths[:, -window:] >= MAX_EPISODE_STEPS).mean() * 100
        lines.append(
            f"{label:23s} | {initial:15.2f} | {final:13.2f} | "
            f"{final_std:10.2f} | {truncated:8.1f}%"
        )

    lines.extend(
        (
            "",
            "Nota: Monte Carlo e TD(0) stimano V(s) per la policy greedy "
            "congelata appresa da Q-Learning; il loro return non rappresenta "
            "un aggiornamento di una policy di controllo.",
        )
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Riepilogo salvato: {out_path}")


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


def print_sensitivity_results():
    """Stampa i punteggi finali delle sensitivity gia' eseguite."""
    for path in sorted(DATA_DIR.glob("acrobot_*_sensitivity_*.npz")):
        saved = np.load(path)
        values = saved["values"]
        curves = saved["curves"]
        scores = curves[:, :, -min(100, curves.shape[-1]) :].mean(axis=(1, 2))
        print(f"\n{path.name}")
        for value, score in zip(values, scores):
            print(f"  {value}: {score:.2f}")


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--window", type=int, default=ROLLING_WINDOW)
    args = parser.parse_args()
    if args.window < 1:
        raise ValueError("--window deve essere positivo")

    data = load_all(args.seeds)
    episodes = validate_data(data)
    window = min(args.window, episodes)

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
    plot_epsilon_decay(
        data, "sarsa_epsilons", "SARSA", "acrobot_sarsa_epsilon.png"
    )

    plot_final_control_performance(
        data, "acrobot_control_final_performance.png", window
    )

    plot_comparison(
        data,
        [("mc_returns", "Monte Carlo"), ("td0_returns", "TD(0)")],
        ylabel="Return",
        title="Return nella prediction (policy greedy congelata) - Acrobot-v1",
        out_name="acrobot_mc_vs_td0_returns.png",
    )

    plot_convergence_comparison(data, "acrobot_mc_vs_td0_convergence.png")
    plot_v_distribution(data, "acrobot_v_distribution.png")

    print_state_coverage(data)
    print_sensitivity_results()
    write_summary(data, window)

    print("\nTutti i grafici sono stati salvati in", FIGURE_DIR)


if __name__ == "__main__":
    main()
