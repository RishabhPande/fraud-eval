"""Detector interface: implement this to plug your own model into the benchmark."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from fraud_eval.attacks import AttackCase


@dataclass
class DetectorResult:
    """What a detector must return for each case."""

    flagged: bool                 # True = detector says "this is fraud/abuse"
    confidence: float = 0.0       # 0.0 - 1.0, optional but useful for ROC-style curves
    label: str = ""               # detector's own category, e.g. "account_takeover"
    latency_ms: float = 0.0       # measured by the Evaluator if not provided
    raw: dict = field(default_factory=dict)  # anything else you want in the report


class Detector(Protocol):
    """Anything with a .decide() method is a detector."""

    def decide(self, case: AttackCase) -> DetectorResult:
        """Return a verdict for one case. Must not raise on weird input."""
        ...


class RuleBasedDetector:
    """Baseline detector: keyword heuristics. Beats nothing, loses to everything.

    Included so `examples/quickstart.py` runs out of the box, and as a
    deliberately weak baseline every real detector should crush.
    """

    SUSPICIOUS = [
        "urgent", "verify your account", "click here", "suspended",
        "wire transfer", "gift card", "otp", "one-time password",
        "recovery code", "sim swap", "crypto", "send money",
    ]

    def decide(self, case: AttackCase) -> DetectorResult:
        start = time.perf_counter()
        text = f"{case.prompt} {case.context}".lower()
        hits = [w for w in self.SUSPICIOUS if w in text]
        elapsed = (time.perf_counter() - start) * 1000
        return DetectorResult(
            flagged=len(hits) >= 2,
            confidence=min(1.0, len(hits) / 4),
            label="heuristic",
            latency_ms=elapsed,
            raw={"keyword_hits": hits},
        )
