# fraud-eval

**The open benchmark for evaluating LLM-based fraud and abuse detectors.**

Every team shipping LLM agents, copilots, and chatbots needs to know: can my
fraud/abuse detector actually catch real attacks — without flagging every
legitimate user? `fraud-eval` gives you a standardized answer: curated
red-team attack packs across the fraud types that actually cost money,
a plug-in interface for any detector, and metrics that matter
(detection rate, false-positive rate, latency, cost per decision).

## Why this exists

Fraud teams at banks, fintechs, and marketplaces are bolting LLM-based
detectors onto account opening, payments, and support flows. There is no
standard benchmark for them — every team hand-rolls its own tests.
`fraud-eval` is that standard.

## Quickstart

```python
from fraud_eval import AttackPack, Evaluator
from fraud_eval.detectors import RuleBasedDetector  # swap in your own

pack = AttackPack.load("account_takeover")          # 40 adversarial cases
detector = RuleBasedDetector()                       # your detector here

results = Evaluator(detector).run(pack)
print(results.summary())
# detection_rate: 0.83 | false_positive_rate: 0.04 | avg_latency_ms: 312
```

## Attack packs

| Pack | Cases | What it tests |
|---|---|---|
| `account_takeover` | 40 | Credential stuffing, SIM-swap social engineering, recovery-flow abuse |
| `phishing_lures` | 40 | Smishing/phishing message generation and detection evasion |
| `synthetic_identity` | 30 | Frankenstein identities, CPN abuse, bust-out patterns |
| `refund_scams` | 30 | Friendly fraud, return abuse, chargeback social engineering |
| `benign_hard` | 50 | Legit users who *look* suspicious — the false-positive killer |

Every pack pairs attack cases with benign twins, because a detector that
flags 90% of attacks and 30% of your customers is a detector that gets
turned off.

## Project layout

```
fraud_eval/
    __init__.py        # public API
    detector.py        # Detector protocol — implement this for your model
    attacks.py         # attack-pack definitions and loaders
    metrics.py         # detection rate, FPR, latency, cost
    runner.py          # the Evaluator
examples/
    quickstart.py
tests/
    test_runner.py
```

## Roadmap

- [x] LLM-as-judge scoring for open-ended detector outputs (`fraud_eval/judge.py` — bring your own LLM via a `complete` callable)
- [ ] Leaderboard: submit your detector's scores
- [ ] Adversarial paraphrase augmentation (attack packs that evolve)
- [ ] Multilingual packs (Hindi, Spanish, Portuguese — fraud is global)
- [ ] Cost-aware metrics: $/correct-decision, not just accuracy

## Contributing

New attack packs welcome — especially from people who fight fraud daily.
Open a PR with your pack: cases + benign twins + a short README on the
threat model.

## License

MIT
