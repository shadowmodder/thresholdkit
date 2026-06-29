# thresholdkit

Choosing the decision threshold is where most fraud / risk / identity models actually live or die. `thresholdkit` turns "what cutoff do we ship?" into a one-liner under the constraint you actually care about.

## Install
```bash
pip install -e .
```

## Usage
```python
from thresholdkit import (best_threshold_for_precision, cost_optimal_threshold,
                          threshold_at_fpr, best_f1_threshold)

y_true  = [0, 0, 0, 1, 1, 1]
y_score = [0.1, 0.4, 0.45, 0.5, 0.8, 0.9]

# Maximize recall while holding precision >= 0.95
best_threshold_for_precision(y_true, y_score, min_precision=0.95)

# Minimize expected cost when a missed fraud (FN) costs 10x a false alarm (FP)
cost_optimal_threshold(y_true, y_score, cost_fp=1.0, cost_fn=10.0)

# Operate at or below a target false-positive rate
threshold_at_fpr(y_true, y_score, target_fpr=0.01)
```

Each call returns a small dict with the chosen `threshold` and the relevant trade-off metrics. MIT licensed.
