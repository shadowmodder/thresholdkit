"""Threshold selection helpers for fraud / risk / IDV style scorers."""
from __future__ import annotations
import numpy as np


def _prep(y_true, y_score):
    return np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float)


def candidate_thresholds(y_score):
    return np.unique(np.asarray(y_score, dtype=float))


def _pr_at(y_true, y_score, t):
    yp = (y_score >= t).astype(int)
    tp = np.sum((yp == 1) & (y_true == 1))
    fp = np.sum((yp == 1) & (y_true == 0))
    fn = np.sum((yp == 0) & (y_true == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(prec), float(rec)


def best_threshold_for_precision(y_true, y_score, min_precision):
    """Lowest-cost operating point meeting a precision floor, maximizing recall."""
    y_true, y_score = _prep(y_true, y_score)
    best = None
    for t in candidate_thresholds(y_score):
        prec, rec = _pr_at(y_true, y_score, t)
        if prec >= min_precision and (best is None or rec > best["recall"]):
            best = {"threshold": float(t), "recall": rec, "precision": prec}
    return best


def best_f1_threshold(y_true, y_score):
    y_true, y_score = _prep(y_true, y_score)
    best = None
    for t in candidate_thresholds(y_score):
        prec, rec = _pr_at(y_true, y_score, t)
        score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if best is None or score > best["f1"]:
            best = {"threshold": float(t), "f1": score, "precision": prec, "recall": rec}
    return best


def cost_optimal_threshold(y_true, y_score, cost_fp, cost_fn):
    """Threshold minimizing expected cost given per-error costs."""
    y_true, y_score = _prep(y_true, y_score)
    best = None
    for t in candidate_thresholds(y_score):
        yp = (y_score >= t).astype(int)
        fp = int(np.sum((yp == 1) & (y_true == 0)))
        fn = int(np.sum((yp == 0) & (y_true == 1)))
        cost = cost_fp * fp + cost_fn * fn
        if best is None or cost < best["expected_cost"]:
            best = {"threshold": float(t), "expected_cost": float(cost), "fp": fp, "fn": fn}
    return best


def threshold_at_fpr(y_true, y_score, target_fpr):
    """Lowest threshold (max recall) keeping false-positive rate <= target."""
    y_true, y_score = _prep(y_true, y_score)
    neg = (y_true == 0)
    N = int(np.sum(neg))
    best = None
    for t in candidate_thresholds(y_score):
        yp = (y_score >= t).astype(int)
        fp = int(np.sum((yp == 1) & neg))
        fpr = fp / N if N > 0 else 0.0
        if fpr <= target_fpr and (best is None or t < best["threshold"]):
            best = {"threshold": float(t), "fpr": fpr}
    return best
