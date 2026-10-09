"""The Evaluator: runs a detector over packs and computes metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from fraud_eval.attacks import AttackPack
from fraud_eval.detector import Detector, DetectorResult
from fraud_eval.metrics import PackMetrics


@dataclass
class CaseResult:
    case_id: str
    label: bool
    result: DetectorResult


@dataclass
class EvaluationResults:
    pack_metrics: list[PackMetrics] = field(default_factory=list)
    case_results: list[CaseResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [m.summary() for m in self.pack_metrics]
        if len(self.pack_metrics) > 1:
            dr = sum(m.detection_rate for m in self.pack_metrics) / len(self.pack_metrics)
            fpr = sum(m.false_positive_rate for m in self.pack_metrics) / len(self.pack_metrics)
            lines.append(f"[overall] avg_detection_rate={dr:.2f} avg_false_positive_rate={fpr:.2f}")
        return "\n".join(lines)

    def misses(self) -> list[CaseResult]:
        """Fraud cases the detector failed to flag — the scary list."""
        return [c for c in self.case_results if c.label and not c.result.flagged]

    def false_alarms(self) -> list[CaseResult]:
        """Benign cases the detector flagged — the churn list."""
        return [c for c in self.case_results if not c.label and c.result.flagged]


class Evaluator:
    def __init__(self, detector: Detector):
        self.detector = detector

    def run(self, pack: AttackPack | list[AttackPack]) -> EvaluationResults:
        packs = [pack] if isinstance(pack, AttackPack) else pack
        results = EvaluationResults()
        for p in packs:
            tp = fp = tn = fn = 0
            latencies: list[float] = []
            confidences: list[float] = []
            for case in p.cases:
                start = time.perf_counter()
                try:
                    verdict = self.detector.decide(case)
                except Exception:
                    verdict = DetectorResult(flagged=False, confidence=0.0, label="error")
                elapsed = (time.perf_counter() - start) * 1000
                if verdict.latency_ms == 0.0:
                    verdict.latency_ms = elapsed
                latencies.append(verdict.latency_ms)
                results.case_results.append(
                    CaseResult(case_id=case.id, label=case.label, result=verdict)
                )
                if case.label and verdict.flagged:
                    tp += 1
                    confidences.append(verdict.confidence)
                elif case.label:
                    fn += 1
                elif verdict.flagged:
                    fp += 1
                else:
                    tn += 1
            n_fraud = tp + fn
            n_benign = fp + tn
            results.pack_metrics.append(
                PackMetrics(
                    pack_name=p.name,
                    n_cases=len(p.cases),
                    n_fraud=n_fraud,
                    n_benign=n_benign,
                    true_positives=tp,
                    false_positives=fp,
                    true_negatives=tn,
                    false_negatives=fn,
                    avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
                    avg_confidence_on_hits=(
                        sum(confidences) / len(confidences) if confidences else 0.0
                    ),
                )
            )
        return results
