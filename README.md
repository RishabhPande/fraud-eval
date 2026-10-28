# fraud-eval

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RishabhPande/fraud-eval/blob/main/examples/quickstart.py)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23695131.svg)](https://doi.org/10.5281/zenodo.23695131)
[![Tests](https://github.com/RishabhPande/fraud-eval/actions/workflows/tests.yml/badge.svg)](https://github.com/RishabhPande/fraud-eval/actions/workflows/tests.yml)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/RishabhPande/fraud-eval)

## Links

- Hugging Face dataset (v0.1.0 smoke test, 20 cases): https://huggingface.co/datasets/RishabhPande/fraud-eval
- Hugging Face leaderboard Space: https://huggingface.co/spaces/RishabhPande/fraud-eval-leaderboard
- Hugging Face counterfactual delay demo: https://huggingface.co/spaces/RishabhPande/fraud-eval-counterfactual-demo
- Archived version, Zenodo v0.1.0: https://doi.org/10.5281/zenodo.23695132
- Concept DOI (all versions): https://doi.org/10.5281/zenodo.23695131
- Metric script: https://github.com/RishabhPande/fraud-eval/blob/main/fraud_eval_metric.py
- Contributing guide: https://github.com/RishabhPande/fraud-eval/blob/main/CONTRIBUTING.md
- Author: Rishabh Pande, ORCID: https://orcid.org/0009-0004-1901-7109

**The open benchmark for evaluating LLM-based fraud and abuse detectors.**

Every team shipping LLM agents, copilots, and chatbots needs to know: can my
fraud/abuse detector actually catch real attacks without flagging every
legitimate user? `fraud-eval` gives you a standardized answer: curated
red-team attack packs across the fraud types that actually cost money,
a plug-in interface for any detector, and metrics that matter
(detection rate, false-positive rate, latency, cost per decision).

## Version note - read this before citing a case count

**v0.1.0 ships 20 cases, verified by running the code in `fraud_eval/attacks.py`.**
An earlier version of this README stated 40 / 40 / 30 / 30 / 50 cases per pack
(190 total). That figure does **not** match the code shipped in v0.1.0.
Do not use the 190 figure for this version. The 190-case benchmark paper
version is separate, pending pack expansion and re-verification. The
Hugging Face dataset for v0.1.0 is labelled smoke test for the same reason:
https://huggingface.co/datasets/RishabhPande/fraud-eval

## Cost-aware metrics

Accuracy is only half the bill. A detector that catches 95% of fraud at
$0.10/decision costs you 100x more than one at $0.001/decision, and at
fraud-scale volume, that dwarfs most accuracy gaps.

Report your spend from `decide()` via `DetectorResult.cost_usd`
(LLM API/token cost per decision; `0.0` = free/unknown), and every pack
report includes **`cost_per_correct_decision`**: total spend divided by
correct calls. See `examples/cost_example.py` for the comparison in action.

## Why this exists

Fraud teams at banks, fintechs, and marketplaces are bolting LLM-based
detectors onto account opening, payments, and support flows. There is no
standard benchmark for them. Every team hand-rolls its own tests.
`fraud-eval` is that standard.

## Quickstart

```python
from fraud_eval import AttackPack, Evaluator
from fraud_eval.detectors import RuleBasedDetector  # swap in your own

pack = AttackPack.load("account_takeover")          # 4 cases in v0.1.0 (3 fraud, 1 benign)
detector = RuleBasedDetector()                       # your detector here

results = Evaluator(detector).run(pack)
print(results.summary())
# [account_takeover] cases=4 detection_rate=0.00 false_positive_rate=0.00 precision=0.00 avg_latency_ms=0
```

## Attack packs

| Pack | Cases | Fraud | Benign | Baseline detection | Baseline FPR | What it tests |
|---|---|---|---|---|---|---|
| `account_takeover` | 4 | 3 | 1 | 0.00 | 0.00 | Credential stuffing, SIM-swap social engineering, recovery-flow abuse |
| `phishing_lures` | 5 | 4 | 1 | 0.25 | 0.00 | Smishing/phishing message generation and detection evasion |
| `synthetic_identity` | 4 | 3 | 1 | 0.00 | 0.00 | Frankenstein identities, CPN abuse, bust-out patterns |
| `refund_scams` | 4 | 3 | 1 | 0.00 | 0.00 | Friendly fraud, return abuse, chargeback social engineering |
| `benign_hard` | 3 | 0 | 3 | - | 0.00 | Legit users who *look* suspicious (the false-positive killer) |
| **Total** | **20** | **13** | **7** | **0.05 pack-average** | **0.00** | |

Every attack pack pairs attack cases with benign twins, because a detector that
flags 90% of attacks and 30% of your customers is a detector that gets
turned off.

## Project layout

```
fraud_eval/
    __init__.py        # public API
    detector.py        # Detector protocol: implement this for your model
    attacks.py         # attack-pack definitions and loaders
    metrics.py         # detection rate, FPR, latency, cost
    runner.py          # the Evaluator
    counterfactual.py  # evaluation under selective + delayed labels
examples/
    quickstart.py
    judge_example.py
    cost_example.py      # cost-aware metrics demo ($/correct-decision)
    counterfactual_example.py  # naive vs corrected net savings
tests/
    test_runner.py
    test_counterfactual.py
```

## Counterfactual evaluation (selective and delayed labels)

`fraud_eval/counterfactual.py` is a clean-room rebuild of the evaluation
protocol from "Counterfactual Evaluation of Fraud Models under Selective
and Delayed Labels": the champion declines high-score orders, so declined
orders never reveal their fraud status, and chargebacks arrive weeks
later, so recent labels are immature. Naive offline metrics inherit both
distortions.

The module provides, in pure stdlib Python:

- `simulate_orders()` - a synthetic order log with known ground truth:
  a deterministic threshold champion, an exploration layer that
  completes declined orders with a logged per-stratum probability, and
  lognormal chargeback delays. No real data anywhere.
- Three net-savings estimators for a candidate flag: `naive_net_savings`
  (visible labels over completed orders), `ipw_net_savings`
  (delay-corrected inverse-propensity weighting, configurable clip cap),
  and `dr_net_savings` (delay-corrected doubly robust, with a weighted
  logistic outcome model). Plus precision at capacity and AUC, and
  `ground_truth_*` helpers for validation against known truth.
- `promotion_gate()` - promote a challenger only when a lower
  confidence bound on its paired advantage over the champion is
  positive. The interval is the normal approximation on paired
  doubly-robust pseudo-values, with a Bonferroni correction when
  several challengers are screened. Intervals ignore
  nuisance-estimation error, so treat them as approximate.
- `partial_identification_bounds()` - the sharp bounds that remain
  when exploration is zero. With no exploration, IPW/DR report
  `identified=False` instead of crashing or silently extrapolating.

Run the demo from the repo root (it reproduces the paper's headline
finding on a fixed seed: naive savings badly biased, corrected savings
close to truth, and a wide identified set with no exploration):

```bash
PYTHONPATH=. python examples/counterfactual_example.py
```

Cite this benchmark as a whole via `CITATION.cff` in the repo root
(GitHub's "Cite this repository" button exports it as BibTeX).

## Roadmap

- [x] LLM-as-judge scoring for open-ended detector outputs (`fraud_eval/judge.py`: bring your own LLM via a `complete` callable)
- [x] Cost-aware metrics: $/correct-decision, not just accuracy
- [x] Counterfactual evaluation under selective and delayed labels (`fraud_eval/counterfactual.py`)
- [ ] Leaderboard: submit your detector's scores
- [ ] Adversarial paraphrase augmentation (attack packs that evolve)
- [ ] Multilingual packs (Hindi, Spanish, Portuguese; fraud is global)

## Contributing

New attack packs welcome, especially from people who fight fraud daily.
Open a PR with your pack: cases + benign twins + a short README on the
threat model.

## License

MIT

## Part of a fraud-ML series

- [GNN Fraud Ring Detection](https://github.com/RishabhPande/GNN-fraud-ring-detection) - GraphSAGE, GAT, GCN, GIN and LSTM from scratch in pure Python (zero dependencies) for fraud ring detection
- [Agentic Fraud Investigator](https://github.com/RishabhPande/agentic-fraud-investigator) - A multi-agent fraud investigation system in pure Python, zero dependencies.
- [ID-document-forgery-detection-CV-](https://github.com/RishabhPande/ID-document-forgery-detection-CV-) - Classic image forensics for ID-document forgery detection, written from scratch in pure Python, zero dependencies.
- [Time-series-anomaly-detection](https://github.com/RishabhPande/Time-series-anomaly-detection) - Time-series anomaly detection, including collective anomalies
- [fraud-rings](https://github.com/RishabhPande/fraud-rings) - Synthetic identity-fraud ring generator plus graph-based detection, all in pure Python, no dependencies.
