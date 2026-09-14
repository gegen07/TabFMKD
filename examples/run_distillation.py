"""Distill a tabular teacher into XGBoost and print a comparison table.

Run from a checkout without installing this package::

    python examples/run_distillation.py --teacher sklearn --dataset breast_cancer
    python examples/run_distillation.py --csv data/table.csv --target label --task classification

Pass ``--teacher tabfm`` or ``--teacher tabicl`` if the matching extra is installed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from tabfm_kd import DistillConfig, TabFMDistiller
from tabfm_kd.data import load_dataset, train_eval_split


def _drop_columns(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [col.strip() for col in raw.split(",") if col.strip()]


def _fmt(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "—"

    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to CSV / TSV / Parquet / Feather instead of a built-in dataset.",
    )
    parser.add_argument("--target", default=None, help="Target column when using --csv.")
    parser.add_argument(
        "--drop",
        default=None,
        help="Comma-separated columns to drop from X (ids, timestamps).",
    )
    parser.add_argument(
        "--task",
        dest="task_type",
        choices=["classification", "regression"],
        default="classification",
        help="Supervised task type when using --csv.",
    )
    parser.add_argument("--teacher", default="sklearn", choices=["tabfm", "tabicl", "sklearn"])
    parser.add_argument("--backend", default="pytorch")
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: auto (CUDA if available), cpu, or cuda.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    bundle = load_dataset(
        args.dataset,
        csv=args.csv,
        target=args.target,
        task_type=args.task_type,
        drop=_drop_columns(args.drop),
    )
    X_train, X_test, y_train, y_test = train_eval_split(
        bundle, test_size=0.25, seed=42
    )

    distiller = TabFMDistiller(
        DistillConfig(
            task_type=bundle.task_type,
            teacher=args.teacher,
            teacher_backend=args.backend,
            teacher_n_estimators=4,
            n_folds=args.n_folds,
            device=args.device,
            teacher_sample_size=150000,
            teacher_sample_strategy="balanced",
            predict_chunk_size=8192,
            temperature=3.0,
            alpha=0.7,
            xgb_params={"n_estimators": 120, "max_depth": 4},
        )
    )
    distiller.fit(X_train, y_train)
    report = distiller.compare(X_test, y_test)
    distiller.save("artifacts/student.joblib")

    if bundle.task_type == "classification":
        keys = [
            "accuracy",
            "f1_macro",
            "roc_auc",
            "gini",
            "average_precision",
            "log_loss",
        ]
        print(f"{'model':<14} " + " ".join(f"{k:>18}" for k in keys))
        for name in ("teacher", "student", "hard_xgb"):
            row = report.get(name) or {}
            print(f"{name:<14} " + " ".join(f"{_fmt(row, k):>18}" for k in keys))
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
