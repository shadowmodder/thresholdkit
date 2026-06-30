from thresholdkit import (best_threshold_for_precision, best_f1_threshold,
                          cost_optimal_threshold, threshold_at_fpr,
                          expected_value_threshold, threshold_table)

YT = [0, 0, 0, 1, 1, 1]
YS = [0.1, 0.4, 0.45, 0.5, 0.8, 0.9]


def test_precision_floor():
    r = best_threshold_for_precision(YT, YS, min_precision=1.0)
    assert r is not None and r["precision"] >= 1.0 and r["recall"] > 0


def test_best_f1():
    r = best_f1_threshold(YT, YS)
    assert 0.0 <= r["f1"] <= 1.0 and "threshold" in r


def test_cost_optimal_prefers_cheaper_errors():
    r = cost_optimal_threshold(YT, YS, cost_fp=1.0, cost_fn=10.0)
    assert "threshold" in r and r["expected_cost"] >= 0


def test_threshold_at_fpr():
    r = threshold_at_fpr(YT, YS, target_fpr=0.0)
    assert r is not None and r["fpr"] <= 0.0 + 1e-9


def test_expected_value_threshold():
    r = expected_value_threshold(YT, YS, reward_tp=10, reward_tn=1, cost_fp=5, cost_fn=8)
    assert r is not None
    assert "threshold" in r and "expected_value" in r
    assert "tp" in r and "fp" in r


def test_threshold_table_coverage():
    rows = threshold_table(YT, YS, step=0.05)
    assert len(rows) == 19  # 0.05 to 0.95 inclusive
    for row in rows:
        assert {"threshold", "precision", "recall", "f1", "fpr", "fp", "fn"} <= set(row)
