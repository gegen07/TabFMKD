"""Distill a tabular teacher into XGBoost and print a comparison table.

Run from a checkout without installing this package::

    python examples/run_distillation.py --teacher sklearn --dataset breast_cancer

Pass ``--teacher tabfm`` on Python >= 3.11 if Google TabFM is already installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from sklearn.model_selection import train_test_split

from tabfm_kd import DistillConfig, TabFMDistiller
from tabfm_kd.data import load_builtin


def _fmt(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "—"

    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--teacher", default="sklearn", choices=["tabfm", "sklearn"])
    parser.add_argument("--backend", default="pytorch")
    parser.add_argument("--n-folds", type=int, default=4)
    args = parser.parse_args()

    bundle = load_builtin(args.dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.25,
        random_state=42,
        stratify=bundle.y if bundle.task_type == "classification" else None,
    )

    distiller = TabFMDistiller(
        DistillConfig(
            task_type=bundle.task_type,
            teacher=args.teacher,
            teacher_backend=args.backend,
            teacher_n_estimators=4,
            n_folds=args.n_folds,
            temperature=3.0,
            alpha=0.7,
            xgb_params={"n_estimators": 120, "max_depth": 4},
        )
    )
    distiller.fit(X_train, y_train)
    report = distiller.compare(X_test, y_test)
    distiller.save("artifacts/student.joblib")

    if bundle.task_type == "classification":
        keys = ["accuracy", "f1_macro", "roc_auc", "log_loss"]
        print(f"{'model':<14} " + " ".join(f"{k:>12}" for k in keys))
        for name in ("teacher", "student", "hard_xgb"):
            row = report.get(name) or {}
            print(f"{name:<14} " + " ".join(f"{_fmt(row, k):>12}" for k in keys))
        print(f"\naccuracy retention vs teacher: {report.get('retention_accuracy', 0):.1%}")
    else:
        keys = ["rmse", "mae", "r2"]
        print(f"{'model':<14} " + " ".join(f"{k:>12}" for k in keys))
        for name in ("teacher", "student", "hard_xgb"):
            row = report.get(name) or {}
            print(f"{name:<14} " + " ".join(f"{_fmt(row, k):>12}" for k in keys))
        print(f"\nR2 retention vs teacher: {report.get('retention_r2', 0):.1%}")

    print("Saved distilled XGBoost student to artifacts/student.joblib")


if __name__ == "__main__":
    main()
