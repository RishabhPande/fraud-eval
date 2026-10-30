"""Plot the counterfactual example: naive vs corrected net savings.

Same simulation, seed and settings as examples/counterfactual_example.py.
Left: net savings per candidate (ground truth vs three estimators).
Right: with zero exploration the estimators cannot identify the answer, so
the honest output is an interval that contains the truth.

All orders are synthetic (fraud_eval.counterfactual.simulate_orders).
Needs matplotlib (optional): pip install matplotlib

    PYTHONPATH=. python examples/plot_counterfactual.py   # docs/counterfactual.png
"""

import os
import random
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fraud_eval.counterfactual import (
    DEFAULT_A_MIN,
    DEFAULT_LAMBDA,
    dr_net_savings,
    fit_delay_law,
    fit_outcome_model,
    flags_from_scores,
    ground_truth_net_savings,
    ipw_net_savings,
    naive_net_savings,
    partial_identification_bounds,
    simulate_orders,
)

N, SEED, DECLINE_RATE, EXPLORATION = 20000, 3, 0.05, 0.20
LAM, A_MIN = DEFAULT_LAMBDA, DEFAULT_A_MIN
BLUE, RED, GREEN, GREY = "#2e86ab", "#e4572e", "#59a14f", "#555555"


def main():
    log = simulate_orders(n=N, seed=SEED, decline_rate=DECLINE_RATE,
                          exploration=EXPLORATION)
    law = fit_delay_law(log)
    model = fit_outcome_model(log, law)
    rng = random.Random(SEED + 999)
    chal_scores = [lg + rng.gauss(0.0, 0.3) for lg in log.logit]
    champ = flags_from_scores(log.champion_score, DECLINE_RATE)
    chal = flags_from_scores(chal_scores, DECLINE_RATE)

    rows = {}
    for name, flags in (("champion", champ), ("challenger", chal)):
        rows[name] = [
            ground_truth_net_savings(log, flags, LAM, A_MIN),
            naive_net_savings(log, flags, LAM, A_MIN).value,
            ipw_net_savings(log, flags, LAM, A_MIN, law).value,
            dr_net_savings(log, flags, LAM, A_MIN, law, model=model).value,
        ]

    log0 = simulate_orders(n=N, seed=SEED, decline_rate=DECLINE_RATE,
                           exploration=0.0)
    law0 = fit_delay_law(log0)
    lo, hi = partial_identification_bounds(log0, chal, LAM, A_MIN, law0)
    truth0 = ground_truth_net_savings(log0, chal, LAM, A_MIN)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.4),
                               gridspec_kw={"width_ratios": [1.6, 1]})
    labels = ["ground truth", "naive", "IPW (corrected)", "doubly robust"]
    colors = ["#333333", RED, BLUE, GREEN]
    w = 0.19
    for i, name in enumerate(rows):
        for j, v in enumerate(rows[name]):
            a.bar(i + (j - 1.5) * w, v, w, color=colors[j],
                  label=labels[j] if i == 0 else None)
            a.text(i + (j - 1.5) * w, v + 0.01, f"{v:.2f}", ha="center",
                   fontsize=8, color=GREY)
    a.set_xticks(range(len(rows)))
    a.set_xticklabels(list(rows))
    a.set_ylabel("net savings (synthetic units)")
    a.set_title("Naive offline savings are badly biased", loc="left")
    a.set_ylim(0, 0.8)
    a.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    a.grid(axis="y", alpha=0.25)

    b.barh([0], [hi - lo], left=[lo], color=BLUE, alpha=0.35, height=0.4)
    b.axvline(truth0, color="#333333", lw=2, label=f"truth {truth0:.2f}")
    b.axvline(0, color="#999999", lw=0.8, ls=":")
    b.set_yticks([])
    b.set_xlabel("challenger net savings")
    b.set_title("No exploration: an interval, not a point", loc="left")
    b.text((lo + hi) / 2, 0.25, f"identified set [{lo:.2f}, {hi:.2f}]",
           ha="center", fontsize=9, color=GREY)
    b.set_ylim(-0.5, 0.6)
    b.set_xlim(-1, 4.3)
    b.legend(frameon=False, fontsize=9, loc="lower right")

    fig.text(0.01, 0.01, "Synthetic order log (fraud_eval.counterfactual."
             "simulate_orders, seed 3, 20,000 orders). Reproduce: "
             "PYTHONPATH=. python examples/plot_counterfactual.py",
             fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(ROOT, "docs", "counterfactual.png")
    fig.savefig(out, dpi=150)
    for k, v in rows.items():
        print(k, [round(x, 3) for x in v])
    print("no-exploration", round(lo, 3), round(hi, 3), round(truth0, 3))


if __name__ == "__main__":
    main()
