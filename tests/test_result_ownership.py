"""A detector may reuse its result object; the evaluator must not rewrite history."""
from unittest.mock import patch

from fraud_eval import AttackPack, Evaluator
from fraud_eval.attacks import AttackCase
from fraud_eval.detector import DetectorResult


def two_cases():
    return AttackPack("reuse", "Synthetic result-ownership checks", [
        AttackCase("fraud", "synthetic fraud", label=True),
        AttackCase("benign", "synthetic benign", label=False),
    ])


class ReusingDetector:
    def __init__(self):
        self.result = DetectorResult(False)

    def decide(self, case):
        self.result.flagged = case.label
        self.result.confidence = 0.9 if case.label else 0.2
        self.result.label = case.id
        self.result.cost_usd = 0.02 if case.label else 0.03
        self.result.raw["case_id"] = case.id
        return self.result


def test_case_results_keep_their_own_scalar_verdicts():
    results = Evaluator(ReusingDetector()).run(two_cases())
    first, second = [c.result for c in results.case_results]
    assert first is not second
    assert (first.flagged, first.confidence, first.label, first.cost_usd) == (
        True, 0.9, "fraud", 0.02)
    assert (second.flagged, second.confidence, second.label, second.cost_usd) == (
        False, 0.2, "benign", 0.03)
    assert results.misses() == []
    assert results.false_alarms() == []


def test_case_results_copy_top_level_raw_entries():
    detector = ReusingDetector()
    results = Evaluator(detector).run(two_cases())
    assert [c.result.raw["case_id"] for c in results.case_results] == ["fraud", "benign"]
    detector.result.raw["later"] = True
    assert all("later" not in c.result.raw for c in results.case_results)


def test_evaluator_does_not_fill_latency_on_detector_owned_result():
    detector = ReusingDetector()
    with patch("fraud_eval.runner.time.perf_counter", side_effect=[1, 1.01, 2, 2.03]):
        results = Evaluator(detector).run(two_cases())
    assert detector.result.latency_ms == 0.0
    latencies = [c.result.latency_ms for c in results.case_results]
    assert abs(latencies[0] - 10) < 1e-8
    assert abs(latencies[1] - 30) < 1e-8
    assert abs(results.pack_metrics[0].avg_latency_ms - 20) < 1e-8


def test_later_detector_mutation_does_not_rewrite_results():
    detector = ReusingDetector()
    results = Evaluator(detector).run(two_cases())
    detector.result.flagged = True
    detector.result.confidence = 0.5
    detector.result.cost_usd = 999
    assert results.false_alarms() == []
    assert results.case_results[1].result.confidence == 0.2
    assert results.pack_metrics[0].total_cost_usd == 0.05
    assert results.case_results[1].result.cost_usd == 0.03


def test_explicit_latency_is_preserved_without_sharing_result():
    class Explicit:
        result = DetectorResult(True, latency_ms=7.5)

        def decide(self, case):
            return self.result

    detector = Explicit()
    results = Evaluator(detector).run(two_cases())
    assert all(c.result.latency_ms == 7.5 for c in results.case_results)
    assert all(c.result is not detector.result for c in results.case_results)
    assert detector.result.latency_ms == 7.5


def test_exception_fallback_still_continues_to_next_case():
    class FailingFirst(ReusingDetector):
        def decide(self, case):
            if case.label:
                raise RuntimeError("synthetic detector failure")
            return super().decide(case)

    results = Evaluator(FailingFirst()).run(two_cases())
    assert len(results.case_results) == 2
    assert results.case_results[0].result.label == "error"
    assert results.pack_metrics[0].false_negatives == 1
    assert results.pack_metrics[0].true_negatives == 1
