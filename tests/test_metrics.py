import numpy as np

from tabfm_kd.metrics import (
    classification_metrics,
    confusion_counts,
    format_confusion_matrix,
    gini_from_auc,
)


def test_gini_is_twice_auc_minus_one():
    assert gini_from_auc(0.5) == 0.0
    assert gini_from_auc(1.0) == 1.0
    assert abs(gini_from_auc(0.75) - 0.5) < 1e-12


def test_imbalanced_metrics_include_average_precision_and_gini():
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    y_proba = np.array(
        [
            [0.95, 0.05],
            [0.90, 0.10],
            [0.85, 0.15],
            [0.80, 0.20],
            [0.70, 0.30],
            [0.60, 0.40],
            [0.45, 0.55],
            [0.20, 0.80],
            [0.10, 0.90],
            [0.55, 0.45],
        ]
    )
    metrics = classification_metrics(y_true, y_pred, y_proba)
    assert "average_precision" in metrics
    assert "gini" in metrics
    assert 0.0 < metrics["average_precision"] <= 1.0
    assert abs(metrics["gini"] - gini_from_auc(metrics["roc_auc"])) < 1e-12
    # Perfect ranking of the three positives would give AP well above prevalence (0.3).
    assert metrics["average_precision"] > 0.3


def test_multiclass_average_precision_is_macro_ovr():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = y_true.copy()
    y_proba = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.1, 0.8],
            [0.1, 0.2, 0.7],
        ]
    )
    metrics = classification_metrics(y_true, y_pred, y_proba)
    assert 0.0 < metrics["average_precision"] <= 1.0
    assert abs(metrics["gini"] - (2.0 * metrics["roc_auc"] - 1.0)) < 1e-12


def test_confusion_matrix_rows_are_true_labels():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    report = confusion_counts(y_true, y_pred)
    assert report["labels"] == [0, 1]
    assert report["matrix"] == [[2, 1], [1, 2]]
    text = format_confusion_matrix(
        report, title="Student (distilled) confusion matrix"
    )
    assert "Student (distilled) confusion matrix" in text
    assert "true 0" in text
    assert "pred 1" in text
