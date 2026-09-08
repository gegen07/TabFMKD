"""Command-line entry point for TabFM → XGBoost distillation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tabfm_kd.data import load_dataset, train_eval_split
from tabfm_kd.distillation import DistillConfig, TabFMDistiller


def _drop_columns(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [col.strip() for col in raw.split(",") if col.strip()]


def _add_shared_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        default="breast_cancer",
        help="Built-in table: breast_cancer, wine, iris, diabetes.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to CSV / TSV / Parquet / Feather instead of --dataset.",
    )
    parser.add_argument("--target", default=None, help="Target column when using --csv.")
    parser.add_argument(
        "--drop",
        default=None,
        help="Comma-separated feature columns to drop (ids, timestamps, leaks).",
    )
    parser.add_argument(
        "--task",
        dest="task_type",
        choices=["classification", "regression"],
        default="classification",
        help="Supervised task type. Used for --csv tables; ignored for built-ins.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)


def _cmd_distill(args: argparse.Namespace) -> int:
    bundle = load_dataset(
        args.dataset,
        csv=args.csv,
        target=args.target,
        task_type=args.task_type,
        drop=_drop_columns(args.drop),
    )
    task_type = bundle.task_type
    X_train, X_test, y_train, y_test = train_eval_split(
        bundle, test_size=args.test_size, seed=args.seed
    )

    config = DistillConfig(
        task_type=task_type,
        teacher=args.teacher,
        teacher_backend=args.backend,
        teacher_n_estimators=args.teacher_estimators,
        teacher_max_num_rows=args.max_num_rows,
        teacher_sample_size=args.teacher_sample_size,
        temperature=args.temperature,
        alpha=args.alpha,
        n_folds=args.n_folds,
        adaptive_temperature=not args.no_adaptive_temperature,
        confidence_weighting=not args.no_confidence_weighting,
        augment_factor=args.augment_factor,
        xgb_params={"n_estimators": args.xgb_rounds},
        random_state=args.seed,
    )
    distiller = TabFMDistiller(config=config)
    distiller.fit(X_train, y_train)
    comparison = distiller.compare(X_test, y_test)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    distiller.save(str(output))

    print(json.dumps(comparison, indent=2))
    print(f"\nSaved student (teacher stripped) to {output}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    bundle = load_dataset(
        args.dataset,
        csv=args.csv,
        target=args.target,
        task_type=args.task_type,
        drop=_drop_columns(args.drop),
    )
    _, X_test, _, y_test = train_eval_split(
        bundle, test_size=args.test_size, seed=args.seed
    )
    distiller = TabFMDistiller.load(args.student)
    metrics = distiller.evaluate(X_test, y_test)
    print(json.dumps(metrics, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tabfm-kd",
        description="Distill Google TabFM into a deployable XGBoost student.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    distill = sub.add_parser("distill", help="Fit TabFM teacher and distill into XGBoost.")
    _add_shared_data_args(distill)
    distill.add_argument("--teacher", default="tabfm", choices=["tabfm", "sklearn"])
    distill.add_argument("--backend", default="pytorch", choices=["pytorch", "jax"])
    distill.add_argument("--teacher-estimators", type=int, default=8)
    distill.add_argument("--max-num-rows", type=int, default=100)
    distill.add_argument(
        "--teacher-sample-size",
        type=int,
        default=None,
        help=(
            "Max rows passed to each teacher fit. Default is "
            "max_num_rows * teacher_estimators. 0 uses the full fold."
        ),
    )
    distill.add_argument("--n-folds", type=int, default=5)
    distill.add_argument("--temperature", type=float, default=3.0)
    distill.add_argument("--alpha", type=float, default=0.7)
    distill.add_argument("--xgb-rounds", type=int, default=200)
    distill.add_argument("--augment-factor", type=float, default=None)
    distill.add_argument("--no-adaptive-temperature", action="store_true")
    distill.add_argument("--no-confidence-weighting", action="store_true")
    distill.add_argument("--output", default="artifacts/student.joblib")
    distill.set_defaults(func=_cmd_distill)

    evaluate = sub.add_parser("evaluate", help="Score a saved student (no TabFM needed).")
    _add_shared_data_args(evaluate)
    evaluate.add_argument("--student", required=True)
    evaluate.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
