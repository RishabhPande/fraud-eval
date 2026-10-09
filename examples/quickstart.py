"""Quickstart: evaluate the built-in baseline on every pack."""

from fraud_eval import AttackPack, Evaluator
from fraud_eval.detector import RuleBasedDetector

detector = RuleBasedDetector()
packs = [AttackPack.load(name) for name in AttackPack.available()]

results = Evaluator(detector).run(packs)
print(results.summary())
print()

misses = results.misses()
print(f"Missed fraud cases ({len(misses)}):")
for m in misses[:5]:
    print(f"  - {m.case_id}")

alarms = results.false_alarms()
print(f"\nFalse alarms ({len(alarms)}):")
for a in alarms[:5]:
    print(f"  - {a.case_id}")

print("\nNow implement your own detector (see fraud_eval/detector.py) and beat this baseline.")
