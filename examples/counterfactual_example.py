"""Counterfactual evaluation example: naive vs corrected net savings.

Simulates an order log with known ground truth, then compares naive
evaluation against delay-corrected IPW and doubly robust estimation for
the champion and one challenger. Headline finding (fixed seed): naive net
savings are visibly biased, while the corrected estimator lands close to
the truth. A second, exploration-free simulation shows the identification
bounds that remain when nothing is explored.

Run:  python examples/counterfactual_example.py
"""

import random

from fraud_eval.counterfactual import (
    DEFAULT_A_MIN,
    DEFAULT_LAMBDA,
    fit_delay_law,
    fit_outcome_model,
    flags_from_scores,
    ground_truth_net_savings,
    ipw_net_savings,
    dr_net_savings,
    naive_net_savings,
    paired_advantage,
    partial_identification_bounds,
    promotion_gate,
    simulate_orders,
)

N = 20000
SEED = 3
DECLINE_RATE = 0.05
EXPLORATION = 0.20
LAM = DEFAULT_LAMBDA
A_MIN = DEFAULT_A_MIN


def main() -> None:
    log = simulate_orders(
        n=N, seed=SEED, decline_rate=DECLINE_RATE, exploration=EXPLORATION
    )
    law = fit_delay_law(log)
    model = fit_outcome_model(log, law)

    # Challenger: same capacity as the champion, but a cleaner score
    # (true logit + small noise instead of the champion's noisier score).
    rng = random.Random(SEED + 999)
    challenger_scores = [lg + rng.gauss(0.0, 0.3) for lg in log.logit]
    champion_flags = flags_from_scores(log.champion_score, DECLINE_RATE)
    challenger_flags = flags_from_scores(challenger_scores, DECLINE_RATE)

    print(f"orders={N} decline_rate={DECLINE_RATE} exploration={EXPLORATION}")
    print(f"target orders (age >= {A_MIN}d): "
          f"{sum(1 for a in log.ages if a >= A_MIN)}")
    print(f"fitted delay median={law.median:.1f}d (true {log.median_delay:.0f}d, "
          f"from {law.n_visible_frauds} visible frauds)")
    print()
    print(f"{'candidate':<12}{'truth':>9}{'naive':>9}{'ipw':>9}{'dr':>9}")
    for name, flags in (("champion", champion_flags),
                        ("challenger", challenger_flags)):
        truth = ground_truth_net_savings(log, flags, LAM, A_MIN)
        naive = naive_net_savings(log, flags, LAM, A_MIN).value
        ipw = ipw_net_savings(log, flags, LAM, A_MIN, law).value
        dr = dr_net_savings(log, flags, LAM, A_MIN, law, model=model).value
        print(f"{name:<12}{truth:9.3f}{naive:9.3f}{ipw:9.3f}{dr:9.3f}")

    adv, se = paired_advantage(
        log, challenger_flags, champion_flags, LAM, A_MIN, law, model=model
    )
    true_adv = (
        ground_truth_net_savings(log, challenger_flags, LAM, A_MIN)
        - ground_truth_net_savings(log, champion_flags, LAM, A_MIN)
    )
    print()
    print(f"challenger advantage: {adv:.3f} (SE {se:.3f}) -> "
          f"promote: {promotion_gate(adv, se)}")
    print(f"true advantage: {true_adv:.3f} (the gate is conservative at "
          f"this volume: a positive point estimate is not enough)")

    # No exploration: nothing in the declined region is identified.
    log_noexp = simulate_orders(
        n=N, seed=SEED, decline_rate=DECLINE_RATE, exploration=0.0
    )
    law_noexp = fit_delay_law(log_noexp)
    lo, hi = partial_identification_bounds(
        log_noexp, challenger_flags, LAM, A_MIN, law_noexp
    )
    truth_noexp = ground_truth_net_savings(log_noexp, challenger_flags, LAM, A_MIN)
    print()
    print(f"no exploration: challenger truth={truth_noexp:.3f} "
          f"identified set=[{lo:.3f}, {hi:.3f}] (width {hi - lo:.3f})")
    print("Without exploration the truth is inside a wide interval, not a point.")


if __name__ == "__main__":
    main()
