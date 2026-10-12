"""Cost-aware evaluation: compare an expensive oracle vs. a cheap heuristic.

Shows why $/correct-decision matters: the expensive detector can win on
accuracy and still lose on economics at scale.
"""

from fraud_eval import AttackPack, Evaluator
from fraud_eval.detector import DetectorResult, RuleBasedDetector


class ExpensiveOracle:
    """Pretend GPT-grade LLM detector: perfect verdicts, $0.02 per call."""

    def decide(self, case):
        return DetectorResult(
            flagged=case.label,
            confidence=0.99,
            label="oracle",
            cost_usd=0.02,
        )


pack = AttackPack.load("account_takeover")

for name, detector in [("expensive_oracle", ExpensiveOracle()),
                       ("rule_baseline", RuleBasedDetector())]:
    results = Evaluator(detector).run(pack)
    m = results.pack_metrics[0]
    print(f"{name:>16}: {m.summary()}")

print("\nSame detection story, very different economics — pick your trade-off.")
