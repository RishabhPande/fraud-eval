from fraud_eval import AttackPack
from fraud_eval.detector import DetectorResult, RuleBasedDetector
from fraud_eval.judge import LLMJudge, summarize


class ScriptedLLM:
    """Deterministic stand-in for a real LLM: grades by keyword sanity.

    NOT a real judge — just lets the pipeline run without API keys.
    A real deployment passes an actual LLM call here.
    """

    def __call__(self, prompt: str) -> str:
        # The prompt embeds the detector output; a smarter stub would parse it.
        # For tests we just exercise the plumbing, so return a fixed grade.
        return "SCORE: 1\nRATIONALE: Stub judge: verdict plausible, reasoning not inspected."


def test_parse_response():
    judge = LLMJudge(ScriptedLLM())
    score, rationale = judge.parse_response(
        "SCORE: 2\nRATIONALE: Correctly identified the SIM-swap pretext."
    )
    assert score == 2
    assert "SIM-swap" in rationale


def test_parse_response_garbage_defaults_to_zero():
    judge = LLMJudge(ScriptedLLM())
    score, _ = judge.parse_response("I cannot comply with that request.")
    assert score == 0


def test_grade_end_to_end():
    judge = LLMJudge(ScriptedLLM())
    case = AttackPack.load("phishing_lures").cases[0]
    result = RuleBasedDetector().decide(case)
    verdict = judge.grade(case, result)
    assert verdict.case_id == case.id
    assert verdict.score == 1
    assert verdict.correct


def test_judge_llm_failure_is_safe():
    def boom(prompt: str) -> str:
        raise RuntimeError("no api key")

    judge = LLMJudge(boom)
    case = AttackPack.load("account_takeover").cases[0]
    verdict = judge.grade(case, DetectorResult(flagged=True))
    assert verdict.score == 0
    assert "failed" in verdict.rationale


def test_summarize():
    from fraud_eval.judge import JudgeVerdict

    verdicts = [
        JudgeVerdict(case_id="a", score=2),
        JudgeVerdict(case_id="b", score=1),
        JudgeVerdict(case_id="c", score=0),
    ]
    s = summarize(verdicts)
    assert s.n_graded == 3
    assert abs(s.avg_score - 1.0) < 1e-9
    assert abs(s.pct_correct_verdict - (2 / 3)) < 1e-9
    assert abs(s.pct_sound_reasoning - (1 / 3)) < 1e-9
    assert "[judge]" in s.summary()
