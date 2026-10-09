"""fraud-eval: the open benchmark for LLM-based fraud detectors."""

from fraud_eval.attacks import AttackCase, AttackPack
from fraud_eval.detector import Detector, DetectorResult, RuleBasedDetector
from fraud_eval.judge import JudgeSummary, JudgeVerdict, LLMJudge, summarize
from fraud_eval.runner import EvaluationResults, Evaluator

__all__ = [
    "AttackCase",
    "AttackPack",
    "Detector",
    "DetectorResult",
    "RuleBasedDetector",
    "EvaluationResults",
    "Evaluator",
    "JudgeSummary",
    "JudgeVerdict",
    "LLMJudge",
    "summarize",
]
__version__ = "0.1.0"
