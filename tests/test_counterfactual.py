"""Tests for fraud_eval.counterfactual.

All simulation tests use fixed seeds, so they are deterministic. Bias
assertions use tolerances and ratios, never exact equality: estimators
are noisy by design, and the claim is that correction shrinks the bias
dramatically, not that it hits the truth exactly.
"""

import math
import random
from dataclasses import replace

from fraud_eval.counterfactual import (
    DEFAULT_A_MIN,
    DelayLaw,
    Estimate,
    auc_score,
    dr_net_savings,
    fit_delay_law,
    flags_from_scores,
    ground_truth_auc,
    ground_truth_net_savings,
    ground_truth_precision,
    ipw_net_savings,
    naive_net_savings,
    paired_advantage,
    partial_identification_bounds,
    precision_at_capacity,
    promotion_gate,
    simulate_orders,
)


def _challenger_flags(log, seed, noise=0.3):
    rng = random.Random(seed + 999)
    scores = [lg + rng.gauss(0.0, noise) for lg in log.logit]
    return flags_from_scores(scores, 0.05)


def test_simulator_structure():
    log = simulate_orders(n=5000, seed=11, decline_rate=0.05, exploration=0.20)
    assert len(log) == 5000
    assert len(log.features[0]) == 10
    assert all(v > 0 for v in log.values)
    # Approved orders always complete with propensity 1.
    for i in range(len(log)):
        if log.champion_flag[i] == 0:
            assert log.completed[i] == 1
            assert log.propensity[i] == 1.0
            assert log.stratum[i] == -1
        else:
            assert log.propensity[i] == 0.20
    # Visible labels are a subset of completed frauds.
    for i in range(len(log)):
        if log.visible[i] == 1:
            assert log.completed[i] == 1
            assert log.fraud[i] == 1
            assert log.delays[i] <= log.ages[i]
    # Decline rate is honored.
    assert abs(sum(log.champion_flag) / len(log) - 0.05) < 0.01


def test_flags_from_scores():
    flags = flags_from_scores([3.0, 1.0, 2.0, 0.0], 0.5)
    assert flags == [1, 0, 1, 0]
    try:
        flags_from_scores([1.0], 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for capacity 0")


def test_delay_law_recovers_median():
    log = simulate_orders(n=10000, seed=3, median_delay=25.0)
    law = fit_delay_law(log)
    assert law.n_visible_frauds >= 20
    # Truncated MLE should land within a factor of 1.5 of the true median.
    assert 25.0 / 1.5 < law.median < 25.0 * 1.5


def test_delay_cdf_floor():
    law = DelayLaw(mu=math.log(25.0), sigma=0.9, n_visible_frauds=10)
    assert law.cdf(0.01) == 0.05
    assert law.cdf(10_000.0) > 0.9


def test_naive_biased_corrected_close():
    # Fixed-seed illustration of the paper's headline finding: naive net
    # savings are badly biased, the corrected estimators land much closer.
    log = simulate_orders(n=20000, seed=3, decline_rate=0.05, exploration=0.20)
    law = fit_delay_law(log)
    flags = _challenger_flags(log, 3)
    truth = ground_truth_net_savings(log, flags)
    naive = naive_net_savings(log, flags).value
    ipw = ipw_net_savings(log, flags, delay_law=law).value
    dr = dr_net_savings(log, flags, delay_law=law).value
    assert abs(naive - truth) > 0.30
    assert abs(ipw - truth) < abs(naive - truth) / 2
    assert abs(dr - truth) < abs(naive - truth) / 2


def test_estimators_ignore_true_labels():
    # Permuting the TRUE fraud labels must not change any estimator:
    # they may only read completed/visible/propensity/ages/values/flags.
    log = simulate_orders(n=8000, seed=5, decline_rate=0.05, exploration=0.20)
    law = fit_delay_law(log)
    flags = _challenger_flags(log, 5)
    shuffled = list(log.fraud)
    random.Random(1).shuffle(shuffled)
    log2 = replace(log, fraud=shuffled)
    assert ipw_net_savings(log, flags, delay_law=law).value == (
        ipw_net_savings(log2, flags, delay_law=law).value
    )
    assert dr_net_savings(log, flags, delay_law=law, max_iter=150).value == (
        dr_net_savings(log2, flags, delay_law=law, max_iter=150).value
    )


def test_zero_exploration_is_unidentified_not_a_crash():
    log = simulate_orders(n=8000, seed=9, decline_rate=0.05, exploration=0.0)
    flags = _challenger_flags(log, 9)
    ipw = ipw_net_savings(log, flags)
    dr = dr_net_savings(log, flags)
    assert ipw.value is None and ipw.identified is False
    assert dr.value is None and dr.identified is False
    # ... but the sharp bounds still exist and bracket the truth.
    law = fit_delay_law(log)
    lo, hi = partial_identification_bounds(log, flags, delay_law=law)
    truth = ground_truth_net_savings(log, flags)
    assert lo <= truth <= hi
    assert hi - lo > 0.5  # wide, as the paper warns


def test_promotion_gate_pure():
    assert promotion_gate(0.30, 0.10) is True
    assert promotion_gate(-0.10, 0.10) is False
    assert promotion_gate(0.05, 0.10) is False  # bound 0.05 - 1.645*0.10 < 0
    # Bonferroni: screening 100 challengers is stricter than screening 1.
    assert promotion_gate(0.30, 0.10, n_challengers=1) is True
    assert promotion_gate(0.30, 0.10, n_challengers=100) is False


def test_gate_holds_for_identical_challenger():
    log = simulate_orders(n=4000, seed=13, decline_rate=0.05, exploration=0.20)
    law = fit_delay_law(log)
    flags = flags_from_scores(log.champion_score, 0.05)
    adv, se = paired_advantage(
        log, flags, flags, delay_law=law, max_iter=100
    )
    assert adv == 0.0
    assert se == 0.0
    assert promotion_gate(adv, se) is False


def test_precision_and_auc_ranges():
    log = simulate_orders(n=8000, seed=17, decline_rate=0.05, exploration=0.20)
    law = fit_delay_law(log)
    flags = _challenger_flags(log, 17)
    for est in (
        precision_at_capacity(log, flags, method="naive"),
        precision_at_capacity(log, flags, delay_law=law, method="ipw"),
    ):
        assert est.value is not None
        assert 0.0 <= est.value <= 1.0
    assert 0.0 <= ground_truth_precision(log, flags) <= 1.0
    scores = log.champion_score
    for est in (
        auc_score(log, scores, method="naive"),
        auc_score(log, scores, method="ipw"),
    ):
        assert est.value is not None
        assert 0.0 <= est.value <= 1.0
    truth_auc = ground_truth_auc(log, scores)
    assert truth_auc is not None and truth_auc > 0.5


def test_clip_caps_weights():
    # Tiny exploration (1%) makes raw weights 100; a clip of 5 must keep
    # the estimate finite and identified rather than exploding.
    log = simulate_orders(n=5000, seed=19, decline_rate=0.05, exploration=0.01)
    law = fit_delay_law(log)
    flags = _challenger_flags(log, 19)
    est = ipw_net_savings(log, flags, delay_law=law, clip=5.0)
    assert est.identified is True
    assert est.value is not None and math.isfinite(est.value)


def test_estimate_dataclass_defaults():
    est = Estimate(1.0, True, "naive", 100)
    assert est.note == ""
