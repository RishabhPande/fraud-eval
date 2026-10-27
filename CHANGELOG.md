# Changelog

All notable changes to fraud-eval will be documented here.

## [0.1.0] - 2026-10-27

### Corrected — 2026-10-27
- **Case counts: 20, not 190.** An earlier version of the README and the
  benchmark paper draft stated 40 / 40 / 30 / 30 / 50 cases per pack
  (190 total). That figure did not match the code shipped in v0.1.0.
  Verified by running `fraud_eval/attacks.py` and the evaluator on
  2026-10-27: account_takeover 4, phishing_lures 5, synthetic_identity 4,
  refund_scams 4, benign_hard 3 — total 20 cases, 13 fraud, 7 benign.
  Do not use the 190 figure for v0.1.0. The 190-case benchmark paper
  version is separate, pending pack expansion and re-verification.
- **Quickstart example removed.** The earlier README quickstart output
  (detection rate 0.83, false-positive rate 0.04, average latency 312 ms)
  did not match any measured run and is removed. The measured
  rule-based baseline is: account_takeover 0.00 / 0.00,
  phishing_lures 0.25 / 0.00, synthetic_identity 0.00 / 0.00,
  refund_scams 0.00 / 0.00, benign_hard false-positive rate 0.00,
  pack-average detection 0.05.

### Added
- Hugging Face dataset (v0.1.0 smoke test, 20 cases):
  https://huggingface.co/datasets/RishabhPande/fraud-eval
- Hugging Face leaderboard Space (smoke test, with results table):
  https://huggingface.co/spaces/RishabhPande/fraud-eval-leaderboard
- Hugging Face counterfactual delay demo:
  https://huggingface.co/spaces/RishabhPande/fraud-eval-counterfactual-demo
- Hugging Face detector arena:
  https://huggingface.co/spaces/RishabhPande/fraud-eval-arena
- Hugging Face curriculum Collection:
  https://huggingface.co/collections/RishabhPande/how-to-evaluate-fraud-detectors
- Metric script (`fraud_eval_metric.py`), contribution guide
  (`CONTRIBUTING.md`), and Colab quickstart notebook
  (`examples/hf_dataset_quickstart.ipynb`).

### Notes
- v0.1.0 is a smoke test. The benchmark paper version (190 cases) is
  not yet released; its counts will be verified by running the code
  before any claim is made.
