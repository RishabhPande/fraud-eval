"""LLM-as-judge scoring for open-ended detector outputs.

Binary flagged/not-flagged metrics (see metrics.py) tell you *whether* a
detector was right. The judge tells you whether it was right *for the right
reasons* — which is what separates a detector you can trust from one that
guesses.

Provider-agnostic: bring your own LLM via a `complete(prompt) -> str`
callable (OpenAI, Anthropic, a local model, anything). No API keys live here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from fraud_eval.attacks import AttackCase
from fraud_eval.detector import DetectorResult


@dataclass
class JudgeVerdict:
    case_id: str
    score: int                    # 2 = right verdict, sound reasoning
                                  # 1 = right verdict, weak/missing reasoning
                                  # 0 = wrong verdict
    rationale: str = ""
    raw_response: str = ""

    @property
    def correct(self) -> bool:
        return self.score >= 1


JUDGE_PROMPT_TEMPLATE = """You are an expert fraud analyst grading an automated fraud detector.
A detector reviewed one case and produced a verdict. Grade it.

CASE UNDER REVIEW:
- Content: {prompt}
- Context: {context}
- Attack type: {attack_type}
- Ground truth: {ground_truth}

DETECTOR OUTPUT:
- Flagged as fraud: {flagged}
- Confidence: {confidence}
- Detector label: {label}
- Detector details: {raw}

Score the detector:
- 2 = correct verdict with sound reasoning evident in its output
- 1 = correct verdict but weak, generic, or missing reasoning
- 0 = wrong verdict

Reply with exactly this format:
SCORE: <0, 1, or 2>
RATIONALE: <one or two sentences explaining your grade>
"""


class LLMJudge:
    """Grades detector outputs with an LLM. You supply the LLM call."""

    def __init__(self, complete: Callable[[str], str]):
        """
        Args:
            complete: a function taking a prompt string and returning the
                LLM's response string. Example with OpenAI:

                from openai import OpenAI
                client = OpenAI()
                judge = LLMJudge(lambda p: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": p}],
                ).choices[0].message.content)
        """
        self.complete = complete

    def build_prompt(self, case: AttackCase, result: DetectorResult) -> str:
        return JUDGE_PROMPT_TEMPLATE.format(
            prompt=case.prompt,
            context=case.context or "(none)",
            attack_type=case.attack_type or "(unknown)",
            ground_truth="FRAUD/ABUSE" if case.label else "BENIGN",
            flagged=result.flagged,
            confidence=f"{result.confidence:.2f}",
            label=result.label or "(none)",
            raw=json.dumps(result.raw) if result.raw else "(none)",
        )

    @staticmethod
    def parse_response(response: str) -> tuple[int, str]:
        score_match = re.search(r"SCORE:\s*([0-2])", response)
        rationale_match = re.search(r"RATIONALE:\s*(.+)", response, re.DOTALL)
        score = int(score_match.group(1)) if score_match else 0
        rationale = rationale_match.group(1).strip() if rationale_match else ""
        return score, rationale

    def grade(self, case: AttackCase, result: DetectorResult) -> JudgeVerdict:
        prompt = self.build_prompt(case, result)
        try:
            response = self.complete(prompt)
        except Exception as e:
            return JudgeVerdict(
                case_id=case.id, score=0,
                rationale=f"judge LLM call failed: {e}",
            )
        score, rationale = self.parse_response(response)
        return JudgeVerdict(
            case_id=case.id, score=score,
            rationale=rationale, raw_response=response,
        )


@dataclass
class JudgeSummary:
    n_graded: int
    avg_score: float                 # 0.0 - 2.0
    pct_correct_verdict: float       # share with score >= 1
    pct_sound_reasoning: float       # share with score == 2
    verdicts: list[JudgeVerdict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[judge] graded={self.n_graded} "
            f"avg_score={self.avg_score:.2f}/2 "
            f"correct_verdict={self.pct_correct_verdict:.0%} "
            f"sound_reasoning={self.pct_sound_reasoning:.0%}"
        )


def summarize(verdicts: list[JudgeVerdict]) -> JudgeSummary:
    n = len(verdicts)
    if n == 0:
        return JudgeSummary(0, 0.0, 0.0, 0.0, [])
    return JudgeSummary(
        n_graded=n,
        avg_score=sum(v.score for v in verdicts) / n,
        pct_correct_verdict=sum(1 for v in verdicts if v.score >= 1) / n,
        pct_sound_reasoning=sum(1 for v in verdicts if v.score == 2) / n,
        verdicts=verdicts,
    )
