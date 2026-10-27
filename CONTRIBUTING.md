# Contributing to fraud-eval: crowdsourced pack expansion

*This guide is published in both places, kept in sync: the Hugging Face dataset card and this GitHub CONTRIBUTING.md.*

## Goal
Grow fraud-eval honestly from the v0.1.0 smoke test (20 cases) toward the 190-case
benchmark paper version, by letting other people contribute packs, with their name
credited in the dataset card for the pack they wrote. A contributor who wrote a pack
will use it, talk about it, and cite it.

## The honest rule (non-negotiable)
- No pack is published as part of the 190-case benchmark until it is expanded,
  reviewed, and re-verified by running the code.
- The v0.1.0 HF dataset stays labelled smoke test, 20 cases, until a new version
  is actually verified. Version numbers follow verified counts, not targets.

## Three levels of contribution
1. **One case, ~5 minutes** — submit a single case in the existing format, as a
   PR to the HF dataset or the GitHub repo. You do not need to write a whole pack.
2. **One pack, credited by name** — submit a full pack (attack cases + benign twins).
   Your name appears in the dataset card under "Pack contributors" for that pack.
3. **One detector, ranked** — submit a detector for the leaderboard Space
   (#1, https://huggingface.co/spaces/RishabhPande/fraud-eval-leaderboard).
   Ranked submissions must be public: detector name, author, and method on the table.

## Case format (must match data/test.jsonl exactly)
One JSON object per line, UTF-8, fields:
- `id`: unique, pack prefix + number, e.g. `ato-005`, `ph-006`
- `pack`: pack name, snake_case, e.g. `account_takeover`
- `prompt`: the user message / content under review (fictional test input only)
- `context`: channel, metadata, conversation history
- `label`: boolean, `true` = fraud/abuse, `false` = benign — never a string
- `attack_type`: e.g. `sim_swap`, `smishing`, `benign`
- `notes`: threat-model notes for contributors

Example (fictional, defensive test input only):
```json
{"id": "ph-006", "pack": "phishing_lures", "prompt": "Fictional example message for detector testing.", "context": "SMS, fictional", "label": true, "attack_type": "smishing", "notes": "Test input to be detected, not a real message."}
```

## Content rules (review will reject on any of these)
- Fictional test prompts only, for detector evaluation. Phishing / scam templates
  are allowed ONLY as defensive test inputs to be detected, labelled as such.
- No real customer data, no Socure data, no real personal data.
- Phone numbers and addresses must be fictional placeholders (555 numbers, etc.).
- Synthetic-identity cases describe generic patterns, not real people.
- Every attack pack except `benign_hard` pairs attack cases with benign twins:
  a detector that flags 90% of attacks and 30% of customers gets turned off, so
  benign twins are what make a pack useful.
- No instructions or templates for wrongdoing. The pack is for detection, not
  for carrying out fraud.

## Review checklist (Kai / Rishabh, before any merge)
- [ ] IDs unique, pack names snake_case, labels boolean
- [ ] No real data, no real PII, placeholders only
- [ ] Benign twins present for attack packs
- [ ] Cases actually run through the repo evaluator without error
- [ ] Count re-verified by running the code, not by counting the PR description
- [ ] Contributor name + pack added to the dataset card, in the same version bump
- [ ] Version note updated: new verified total, never the 190 target until reached

## How to submit
- HF dataset: open a Discussion or PR on https://huggingface.co/datasets/RishabhPande/fraud-eval
- GitHub: PR to https://github.com/RishabhPande/fraud-eval, following the repo's
  AttackCase format in fraud_eval/attacks.py
- Include: your name as you want it credited, pack name, case count, and a one-line
  description of what the pack tests.

## What happens after acceptance
- Your pack ships in the next verified dataset version, with your credit in the card.
- Your pack is added to the leaderboard test set in that version, so detectors are
  scored on it.
- You are free to cite the dataset version that includes your pack, and to link
  your pack in your own profile / papers.

