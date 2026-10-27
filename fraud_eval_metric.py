"""fraud-eval metric — DRAFT, local prep only, NOT published.

Computes the same metrics as fraud_eval/metrics.py PackMetrics, for any
predictions/references pair. No data, no hidden test set, no 190-case claim.
With 20 cases, one case moves a pack rate by 20-33 points.
"""


def compute(predictions, references, costs=None):
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("empty predictions")
    tp = sum(1 for p, r in zip(predictions, references) if p and r)
    fp = sum(1 for p, r in zip(predictions, references) if p and not r)
    fn = sum(1 for p, r in zip(predictions, references) if not p and r)
    tn = sum(1 for p, r in zip(predictions, references) if not p and not r)
    fraud = tp + fn
    benign = fp + tn
    result = {
        "detection_rate": tp / fraud if fraud else 0.0,
        "false_positive_rate": fp / benign if benign else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "correct_decisions": tp + tn,
        "total_cases": len(predictions),
    }
    if costs is not None:
        total_cost = sum(costs)
        result["total_cost_usd"] = total_cost
        result["cost_per_correct_decision"] = (
            total_cost / result["correct_decisions"] if result["correct_decisions"] else 0.0
        )
    return result


if __name__ == "__main__":
    # Tiny synthetic example only, no real data.
    print(compute([True, False], [True, False]))
