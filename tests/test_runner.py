from fraud_eval import AttackPack, Evaluator
from fraud_eval.detector import DetectorResult, RuleBasedDetector


def test_packs_load():
    packs = [AttackPack.load(n) for n in AttackPack.available()]
    assert len(packs) == 5
    assert all(len(p.cases) > 0 for p in packs)


def test_unknown_pack_raises():
    try:
        AttackPack.load("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_baseline_runs():
    pack = AttackPack.load("account_takeover")
    results = Evaluator(RuleBasedDetector()).run(pack)
    assert len(results.pack_metrics) == 1
    m = results.pack_metrics[0]
    assert m.n_cases == len(pack.cases)
    assert 0.0 <= m.detection_rate <= 1.0
    assert 0.0 <= m.false_positive_rate <= 1.0


def test_detector_errors_dont_crash_runner():
    class Exploding:
        def decide(self, case):
            raise RuntimeError("boom")

    results = Evaluator(Exploding()).run(AttackPack.load("benign_hard"))
    m = results.pack_metrics[0]
    assert m.true_positives == 0
    assert m.false_negatives + m.true_negatives == m.n_cases


def test_overall_summary_multi_pack():
    packs = [AttackPack.load(n) for n in AttackPack.available()]
    results = Evaluator(RuleBasedDetector()).run(packs)
    summary = results.summary()
    assert "[overall]" in summary
    assert "cost_per_correct_decision" in summary


class CostedDetector:
    """Stub LLM-style detector with a fixed per-decision price tag."""

    def __init__(self, cost_usd: float):
        self.cost_usd = cost_usd

    def decide(self, case):
        return DetectorResult(
            flagged=case.label,  # perfect oracle, for math checks
            confidence=1.0,
            cost_usd=self.cost_usd,
        )


def test_costs_accumulate_and_per_correct_decision():
    pack = AttackPack.load("phishing_lures")
    results = Evaluator(CostedDetector(cost_usd=0.02)).run(pack)
    m = results.pack_metrics[0]
    assert abs(m.total_cost_usd - len(pack.cases) * 0.02) < 1e-9
    expected = m.total_cost_usd / (m.true_positives + m.true_negatives)
    assert abs(m.cost_per_correct_decision - expected) < 1e-9
    assert "cost_per_correct_decision" in m.summary()


def test_free_detector_costs_zero():
    results = Evaluator(RuleBasedDetector()).run(AttackPack.load("phishing_lures"))
    m = results.pack_metrics[0]
    assert m.total_cost_usd == 0.0
    assert m.cost_per_correct_decision == 0.0


def test_cost_per_correct_decision_no_correct_decisions():
    from fraud_eval.metrics import PackMetrics

    m = PackMetrics(
        pack_name="empty", n_cases=0, n_fraud=0, n_benign=0,
        true_positives=0, false_positives=0, true_negatives=0,
        false_negatives=0, avg_latency_ms=0.0,
        avg_confidence_on_hits=0.0, total_cost_usd=5.0,
    )
    assert m.cost_per_correct_decision == 0.0
