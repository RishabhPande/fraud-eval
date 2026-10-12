"""Metrics that matter for fraud detectors.

Accuracy alone lies: a detector that flags 100% of attacks and 30% of
legitimate users will get turned off on day one. These metrics keep
both sides of the trade-off honest.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PackMetrics:
    pack_name: str
    n_cases: int
    n_fraud: int
    n_benign: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    avg_latency_ms: float
    avg_confidence_on_hits: float
    total_cost_usd: float = 0.0   # summed from DetectorResult.cost_usd

    @property
    def detection_rate(self) -> float:          # recall on fraud cases
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def false_positive_rate(self) -> float:     # FPR on benign cases
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def correct_decisions(self) -> int:
        return self.true_positives + self.true_negatives

    @property
    def cost_per_correct_decision(self) -> float:
        """$/correct-decision: what each right call actually costs you.

        A detector with 95% detection at $0.10/decision may be worse
        business than 90% at $0.001/decision — this metric makes the
        trade-off explicit. 0.0 when there were no correct decisions.
        """
        return self.total_cost_usd / self.correct_decisions if self.correct_decisions else 0.0

    def summary(self) -> str:
        return (
            f"[{self.pack_name}] cases={self.n_cases} "
            f"detection_rate={self.detection_rate:.2f} "
            f"false_positive_rate={self.false_positive_rate:.2f} "
            f"precision={self.precision:.2f} "
            f"avg_latency_ms={self.avg_latency_ms:.0f} "
            f"cost_per_correct_decision=${self.cost_per_correct_decision:.4f}"
        )
