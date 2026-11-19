"""Caught detector failures must be visible without redefining existing metrics."""
from dataclasses import asdict

import pytest

from fraud_eval import AttackPack, Evaluator
from fraud_eval.attacks import AttackCase
from fraud_eval.detector import DetectorResult
from fraud_eval.metrics import PackMetrics
from fraud_eval.runner import CaseResult


def pack(name="mixed", labels=(True, False)):
    return AttackPack(name, "Synthetic exception-reporting regression", [
        AttackCase(f"{name}-{i}", "synthetic input", label=label)
        for i, label in enumerate(labels)
    ])


class Exploding:
    def decide(self, case):
        raise RuntimeError("synthetic private detail must not be retained")


@pytest.mark.parametrize("labels,fn,tn", [((True,), 1, 0), ((False,), 0, 1), ((True, False), 1, 1)])
def test_errors_are_counted_without_changing_fallback_confusion_counts(labels, fn, tn):
    results = Evaluator(Exploding()).run(pack(labels=labels))
    metric = results.pack_metrics[0]
    assert (metric.n_errors, metric.error_rate) == (len(labels), 1.0)
    assert (metric.false_negatives, metric.true_negatives) == (fn, tn)
    assert metric.true_positives == metric.false_positives == 0
    assert results.errors() == results.case_results
    assert all(case.detector_error and case.result.label == "error" for case in results.errors())
    assert f"errors={len(labels)}/{len(labels)}" in results.summary()


def test_returned_error_category_is_not_a_caught_exception():
    class Category:
        def decide(self, case):
            return DetectorResult(flagged=True, label="error")

    results = Evaluator(Category()).run(pack())
    assert results.errors() == []
    assert results.pack_metrics[0].n_errors == 0
    assert results.pack_metrics[0].error_rate == 0.0
    assert all(not case.detector_error for case in results.case_results)


def test_continues_after_failure_and_retains_case_order():
    class Sometimes:
        def decide(self, case):
            if case.label:
                raise ValueError("synthetic failure")
            return DetectorResult(flagged=False, cost_usd=0.02)

    results = Evaluator(Sometimes()).run(pack(labels=(False, True, False, True)))
    assert [case.case_id for case in results.errors()] == ["mixed-1", "mixed-3"]
    metric = results.pack_metrics[0]
    assert (metric.n_errors, metric.error_rate, metric.total_cost_usd) == (2, 0.5, 0.04)
    assert len(results.misses()) == 2
    assert len(results.case_results) == 4


def test_multi_pack_errors_are_case_weighted_not_pack_averaged():
    class Mixed:
        def decide(self, case):
            if case.id.startswith("fail"):
                raise RuntimeError("synthetic failure")
            return DetectorResult(False)

    results = Evaluator(Mixed()).run([
        pack("fail", (False,)), pack("pass", (False,) * 9), pack("empty", ()),
    ])
    assert [metric.n_errors for metric in results.pack_metrics] == [1, 0, 0]
    assert [metric.error_rate for metric in results.pack_metrics] == [1.0, 0.0, 0.0]
    assert "[overall]" in results.summary()
    assert "errors=1/10" in results.summary().splitlines()[-1]
    assert len(results.errors()) == 1


def test_empty_pack_has_zero_errors_and_no_division_by_zero():
    results = Evaluator(Exploding()).run(pack(labels=()))
    metric = results.pack_metrics[0]
    assert (metric.n_errors, metric.error_rate) == (0, 0.0)
    assert results.errors() == []
    assert "errors=0/0" in results.summary()


def test_empty_pack_list_still_has_empty_summary():
    results = Evaluator(Exploding()).run([])
    assert results.errors() == []
    assert results.summary() == ""


def test_legacy_dataclass_construction_defaults_to_no_errors():
    metric = PackMetrics("legacy", 2, 1, 1, 1, 0, 1, 0, 0.0, 0.0, 0.02)
    case = CaseResult("legacy-case", True, DetectorResult(True))
    assert (metric.n_errors, metric.error_rate) == (0, 0.0)
    assert case.detector_error is False


def test_exception_text_is_not_kept_in_serialized_results():
    results = Evaluator(Exploding()).run(pack())
    serialized = asdict(results)
    assert "synthetic private detail" not in repr(serialized)
    assert serialized["case_results"][0]["detector_error"] is True


def test_error_count_resets_between_evaluator_runs():
    evaluator = Evaluator(Exploding())
    first = evaluator.run(pack(labels=(True, False)))
    second = evaluator.run(pack(labels=(False,)))
    assert first.pack_metrics[0].n_errors == 2
    assert second.pack_metrics[0].n_errors == 1
    assert len(first.errors()) == 2
    assert len(second.errors()) == 1
