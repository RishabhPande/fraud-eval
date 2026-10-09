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

    def summary(self) -> str:
        return (
            f"[{self.pack_name}] cases={self.n_cases} "
            f"detection_rate={self.detection_rate:.2f} "
            f"false_positive_rate={self.false_positive_rate:.2f} "
            f"precision={self.precision:.2f} "
            f"avg_latency_ms={self.avg_latency_ms:.0f}"
        )
