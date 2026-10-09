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
