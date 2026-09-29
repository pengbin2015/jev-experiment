"""Evaluate classification predictions against hidden ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GROUND_TRUTH = (
    ROOT
    / "data"
    / "prepared"
    / "test_600_ground_truth.csv"
)

METRICS_DIR = ROOT / "results" / "metrics"

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate model predictions against "
            "the frozen test ground truth."
        )
    )

    parser.add_argument(
        "--predictions",
        required=True,
        help=(
            "Prediction CSV containing "
            "'id' and 'prediction'."
        ),
    )

    parser.add_argument(
        "--label-column",
        required=True,
        choices=[
            "v1_label",
            "v2_label",
        ],
        help=(
            "Ground-truth label column "
            "to evaluate against."
        ),
    )

    parser.add_argument(
        "--name",
        required=True,
        help=(
            "Experiment/model name used "
            "for output filenames."
        ),
    )

    parser.add_argument(
        "--ground-truth",
        default=str(DEFAULT_GROUND_TRUTH),
        help=(
            "Ground-truth CSV. "
            "Defaults to frozen 600-test set."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main evaluation
# ============================================================

def main() -> None:
    args = parse_args()

    predictions_file = Path(
        args.predictions
    )

    ground_truth_file = Path(
        args.ground_truth
    )


    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    pred_df = pd.read_csv(
        predictions_file
    )

    truth_df = pd.read_csv(
        ground_truth_file
    )


    # --------------------------------------------------------
    # Validate prediction format
    # --------------------------------------------------------

    required_prediction_columns = {
        "id",
        "prediction",
    }

    missing = (
        required_prediction_columns
        - set(pred_df.columns)
    )

    if missing:
        raise ValueError(
            "Prediction file missing columns: "
            f"{sorted(missing)}"
        )


    if pred_df["id"].duplicated().any():
        raise ValueError(
            "Prediction file contains "
            "duplicate IDs."
        )


    if truth_df["id"].duplicated().any():
        raise ValueError(
            "Ground-truth file contains "
            "duplicate IDs."
        )


    # --------------------------------------------------------
    # Join predictions with ground truth
    #
    # Ground truth becomes visible ONLY here.
    # --------------------------------------------------------

    evaluation_df = truth_df[
        [
            "id",
            args.label_column,
        ]
    ].merge(
        pred_df[
            [
                "id",
                "prediction",
            ]
        ],
        on="id",
        how="left",
        validate="one_to_one",
    )


    # --------------------------------------------------------
    # Check that every test example received a prediction
    # --------------------------------------------------------

    missing_predictions = (
        evaluation_df["prediction"]
        .isna()
        .sum()
    )

    if missing_predictions:
        raise ValueError(
            f"{missing_predictions} test examples "
            "have no prediction."
        )


    extra_ids = (
        set(pred_df["id"])
        - set(truth_df["id"])
    )

    if extra_ids:
        raise ValueError(
            f"Prediction file contains "
            f"{len(extra_ids)} unknown IDs."
        )


    y_true = evaluation_df[
        args.label_column
    ]

    y_pred = evaluation_df[
        "prediction"
    ]


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )


    # --------------------------------------------------------
    # Per-class report
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    labels = sorted(
        y_true.unique()
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    confusion_df.index.name = "actual"
    confusion_df.columns.name = "predicted"


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    metrics_file = (
        METRICS_DIR
        / f"{args.name}.json"
    )

    report_file = (
        METRICS_DIR
        / f"{args.name}_per_class.csv"
    )

    confusion_file = (
        METRICS_DIR
        / f"{args.name}_confusion.csv"
    )


    metrics = {
        "model": args.name,
        "label_column": args.label_column,
        "test_examples": int(
            len(evaluation_df)
        ),
        "macro_f1": float(
            macro_f1
        ),
        "accuracy": float(
            accuracy
        ),
    }


    with metrics_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )


    report_df.to_csv(
        report_file
    )

    confusion_df.to_csv(
        confusion_file
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"EVALUATION — {args.name}"
    )
    print("=" * 70)

    print(
        f"Test examples : {len(evaluation_df):,}"
    )

    print(
        f"Macro-F1      : {macro_f1:.4f}"
    )

    print(
        f"Accuracy      : {accuracy:.4f}"
    )


    print()
    print("Per-class performance")
    print("-" * 70)

    class_rows = report_df.loc[
        labels,
        [
            "precision",
            "recall",
            "f1-score",
            "support",
        ],
    ]

    print(
        class_rows.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )


    print()
    print("Files created")
    print("-" * 70)

    print(
        f"  results/metrics/{args.name}.json"
    )

    print(
        f"  results/metrics/"
        f"{args.name}_per_class.csv"
    )

    print(
        f"  results/metrics/"
        f"{args.name}_confusion.csv"
    )


if __name__ == "__main__":
    main()