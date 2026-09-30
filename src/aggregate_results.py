"""Aggregate all V2 supervised ML experiment results."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = ROOT / "results" / "metrics"
OUTPUT_DIR = ROOT / "results" / "summary"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Expected metric filename format:
#
# tfidf_lr_v2_b20_d01.json
# embedding_lr_v2_b100_d05.json
# ============================================================

FILENAME_PATTERN = re.compile(
    r"^(?P<model>tfidf_lr_v2|embedding_lr_v2)"
    r"_b(?P<budget>\d+)"
    r"_d(?P<draw>\d+)"
    r"\.json$"
)


MODEL_DISPLAY_NAMES = {
    "tfidf_lr_v2": "TF-IDF + LR",
    "embedding_lr_v2": "Embedding + LR",
}


EXPECTED_MODELS = {
    "tfidf_lr_v2",
    "embedding_lr_v2",
}

EXPECTED_BUDGETS = {
    20,
    100,
}

EXPECTED_DRAWS = {
    1,
    2,
    3,
    4,
    5,
}


# ============================================================
# Discover result files
# ============================================================

def discover_results() -> list[dict]:
    rows = []

    for file in sorted(
        METRICS_DIR.glob("*_v2_b*_d*.json")
    ):
        match = FILENAME_PATTERN.match(
            file.name
        )

        if not match:
            continue

        model = match.group("model")
        budget = int(
            match.group("budget")
        )
        draw = int(
            match.group("draw")
        )

        with file.open(
            "r",
            encoding="utf-8",
        ) as f:
            metrics = json.load(f)

        rows.append(
            {
                "model": model,
                "model_display": (
                    MODEL_DISPLAY_NAMES.get(
                        model,
                        model,
                    )
                ),
                "budget": budget,
                "draw": draw,
                "macro_f1": float(
                    metrics["macro_f1"]
                ),
                "accuracy": float(
                    metrics["accuracy"]
                ),
                "source_file": file.name,
            }
        )

    return rows


# ============================================================
# Validate experiment completeness
# ============================================================

def validate_completeness(
    results_df: pd.DataFrame,
) -> None:

    missing = []

    for model in EXPECTED_MODELS:
        for budget in EXPECTED_BUDGETS:
            for draw in EXPECTED_DRAWS:

                match = results_df[
                    (results_df["model"] == model)
                    & (
                        results_df["budget"]
                        == budget
                    )
                    & (
                        results_df["draw"]
                        == draw
                    )
                ]

                if len(match) == 0:
                    missing.append(
                        (
                            model,
                            budget,
                            draw,
                        )
                    )

                elif len(match) > 1:
                    raise ValueError(
                        "Duplicate result found for "
                        f"{model}, budget={budget}, "
                        f"draw={draw}"
                    )

    if missing:
        print()
        print(
            "WARNING: experiment is not yet complete."
        )
        print(
            "Missing result files:"
        )

        for model, budget, draw in missing:
            print(
                "  "
                f"{model}_"
                f"b{budget}_"
                f"d{draw:02d}.json"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    rows = discover_results()

    if not rows:
        raise FileNotFoundError(
            "No V2 result files found in "
            "results/metrics/"
        )

    results_df = pd.DataFrame(
        rows
    )

    results_df = results_df.sort_values(
        [
            "model",
            "budget",
            "draw",
        ]
    ).reset_index(drop=True)


    # ========================================================
    # Validate completeness
    # ========================================================

    validate_completeness(
        results_df
    )


    # ========================================================
    # Display per-draw results
    # ========================================================

    print()
    print("=" * 90)
    print("V2 PER-DRAW RESULTS")
    print("=" * 90)

    display_columns = [
        "model_display",
        "budget",
        "draw",
        "macro_f1",
        "accuracy",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # ========================================================
    # Aggregate by model + label budget
    # ========================================================

    summary_df = (
        results_df
        .groupby(
            [
                "model",
                "model_display",
                "budget",
            ],
            as_index=False,
        )
        .agg(
            draws=(
                "draw",
                "count",
            ),

            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),

            macro_f1_std=(
                "macro_f1",
                "std",
            ),

            macro_f1_min=(
                "macro_f1",
                "min",
            ),

            macro_f1_max=(
                "macro_f1",
                "max",
            ),

            accuracy_mean=(
                "accuracy",
                "mean",
            ),

            accuracy_std=(
                "accuracy",
                "std",
            ),

            accuracy_min=(
                "accuracy",
                "min",
            ),

            accuracy_max=(
                "accuracy",
                "max",
            ),
        )
        .sort_values(
            [
                "model",
                "budget",
            ]
        )
        .reset_index(drop=True)
    )


    # ========================================================
    # Display aggregate summary
    # ========================================================

    print()
    print("=" * 90)
    print("V2 AGGREGATED RESULTS")
    print("=" * 90)

    summary_display = summary_df[
        [
            "model_display",
            "budget",
            "draws",
            "macro_f1_mean",
            "macro_f1_std",
            "macro_f1_min",
            "macro_f1_max",
            "accuracy_mean",
            "accuracy_std",
            "accuracy_min",
            "accuracy_max",
        ]
    ]

    print(
        summary_display.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # ========================================================
    # Additional label-efficiency view
    #
    # This is useful for the blog:
    #
    # model
    # 20 labels/class  -> quality
    # 100 labels/class -> quality
    # ========================================================

    label_efficiency_df = summary_df[
        [
            "model_display",
            "budget",
            "macro_f1_mean",
            "macro_f1_min",
            "macro_f1_max",
            "accuracy_mean",
        ]
    ].copy()

    label_efficiency_df["new_labels"] = (
        label_efficiency_df["budget"]
        * 6
    )

    label_efficiency_df = (
        label_efficiency_df[
            [
                "model_display",
                "budget",
                "new_labels",
                "macro_f1_mean",
                "macro_f1_min",
                "macro_f1_max",
                "accuracy_mean",
            ]
        ]
        .sort_values(
            [
                "model_display",
                "budget",
            ]
        )
    )


    print()
    print("=" * 90)
    print("LABEL-EFFICIENCY VIEW")
    print("=" * 90)

    print(
        label_efficiency_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


    # ========================================================
    # Save results
    # ========================================================

    per_draw_file = (
        OUTPUT_DIR
        / "v2_ml_per_draw.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "v2_ml_summary.csv"
    )

    label_efficiency_file = (
        OUTPUT_DIR
        / "v2_ml_label_efficiency.csv"
    )

    summary_json_file = (
        OUTPUT_DIR
        / "v2_ml_summary.json"
    )


    results_df.to_csv(
        per_draw_file,
        index=False,
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    label_efficiency_df.to_csv(
        label_efficiency_file,
        index=False,
    )

    with summary_json_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary_df.to_dict(
                orient="records"
            ),
            f,
            indent=2,
        )


    # ========================================================
    # Report
    # ========================================================

    print()
    print("Files created")
    print("-" * 90)

    for file in [
        per_draw_file,
        summary_file,
        label_efficiency_file,
        summary_json_file,
    ]:
        print(
            f"  {file.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()