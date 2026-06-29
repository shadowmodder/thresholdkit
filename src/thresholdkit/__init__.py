"""thresholdkit: pick operating thresholds for binary scorers under real constraints."""
from .core import (best_threshold_for_precision, best_f1_threshold,
                   cost_optimal_threshold, threshold_at_fpr, candidate_thresholds)

__all__ = ["best_threshold_for_precision", "best_f1_threshold",
           "cost_optimal_threshold", "threshold_at_fpr", "candidate_thresholds"]
__version__ = "0.1.0"
