"""Judge example: grade the baseline detector with an LLM judge.

Swap ScriptedLLM for a real LLM call (see fraud_eval/judge.py docstring)
to get meaningful grades.
"""

from fraud_eval import AttackPack, Evaluator
from fraud_eval.detector import RuleBasedDetector
from fraud_eval.judge import LLMJudge, summarize


class ScriptedLLM:
    """Placeholder judge. Replace with a real LLM for real grades."""

    def __call__(self, prompt: str) -> str:
        return "SCORE: 1\nRATIONALE: Stub judge: verdict plausible, reasoning not inspected."


pack = AttackPack.load("phishing_lures")
results = Evaluator(RuleBasedDetector()).run(pack)
print(results.summary())

cases = {c.id: c for c in pack.cases}
judge = LLMJudge(ScriptedLLM())
verdicts = [
    judge.grade(cases[cr.case_id], cr.result) for cr in results.case_results
]
print(summarize(verdicts).summary())
