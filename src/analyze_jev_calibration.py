"""Analyze Jev probability calibration and confidence as a risk signal."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    ROOT
    / "results"
    / "raw"
    / "jev_zero_shot_v2_raw.csv"
)

GROUND_TRUTH_FILE = (
    ROOT
    / "data"
    / "prepared"
    / "test_600_ground_truth.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "calibration"
)

FIGURES_DIR = (
    ROOT
    / "results"
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

NUM_ECE_BINS = 10
NUM_CONFIDENCE_QUANTILES = 10

REVIEW_FRACTIONS = [
    0.10,
    0.20,
    0.30,
]


LABELS = [
    "ACCOUNT_SERVICES",
    "CARD_MANAGEMENT",
    "CASH_WITHDRAWAL",
    "CURRENCY",
    "DIGITAL_PAYMENTS",
    "IDENTITY_COMPLIANCE",
    "SECURITY_DISPUTES",
    "TOP_UPS",
    "TRANSFERS",
]


# ============================================================
# Load data
# ============================================================

raw_df = pd.read_csv(
    RAW_FILE
)

truth_df = pd.read_csv(
    GROUND_TRUTH_FILE
)[
    [
        "id",
        "v2_label",
    ]
]


df = raw_df.merge(
    truth_df,
    on="id",
    how="inner",
    validate="one_to_one",
)


if len(df) != 600:
    raise ValueError(
        f"Expected 600 examples, got {len(df)}."
    )


if df["id"].duplicated().any():
    raise ValueError(
        "Duplicate IDs found after merge."
    )


unknown_predictions = (
    set(df["prediction"])
    - set(LABELS)
)

if unknown_predictions:
    raise ValueError(
        "Unexpected prediction labels: "
        f"{sorted(unknown_predictions)}"
    )


unknown_truth = (
    set(df["v2_label"])
    - set(LABELS)
)

if unknown_truth:
    raise ValueError(
        "Unexpected ground-truth labels: "
        f"{sorted(unknown_truth)}"
    )


# ============================================================
# Parse Jev probability dictionaries
# ============================================================

probability_dicts = []

for value in df["probabilities"]:

    parsed = json.loads(
        value
    )

    probability_dicts.append(
        parsed
    )


raw_prob_matrix = np.array(
    [
        [
            float(
                probs.get(
                    label,
                    0.0,
                )
            )
            for label in LABELS
        ]
        for probs in probability_dicts
    ],
    dtype=float,
)


# ============================================================
# Validate raw probability vectors
# ============================================================

raw_probability_sums = (
    raw_prob_matrix.sum(
        axis=1
    )
)


if np.any(
    raw_probability_sums <= 0
):
    raise ValueError(
        "At least one Jev response has "
        "a non-positive probability sum."
    )


max_probability_sum_error = float(
    np.max(
        np.abs(
            raw_probability_sums
            - 1.0
        )
    )
)


# ============================================================
# Normalize probability vectors
#
# TypeSafe documents that returned probabilities may sum
# approximately to 1. We normalize before Brier/log-loss
# calculations so they operate on proper distributions.
# ============================================================

prob_matrix = (
    raw_prob_matrix
    / raw_probability_sums[
        :,
        np.newaxis,
    ]
)


normalized_probability_sums = (
    prob_matrix.sum(
        axis=1
    )
)


max_normalized_sum_error = float(
    np.max(
        np.abs(
            normalized_probability_sums
            - 1.0
        )
    )
)


# ============================================================
# Correct / incorrect classification
# ============================================================

df["correct"] = (
    df["prediction"]
    == df["v2_label"]
).astype(int)


accuracy = float(
    df["correct"].mean()
)

total_errors = int(
    (
        df["correct"] == 0
    ).sum()
)


# ============================================================
# Selected-class probabilities
#
# Keep raw and normalized probabilities separately.
# ============================================================

label_to_index = {
    label: index
    for index, label
    in enumerate(LABELS)
}


selected_raw_probabilities = []
selected_normalized_probabilities = []


for row_index, prediction in enumerate(
    df["prediction"]
):

    class_index = (
        label_to_index[
            prediction
        ]
    )

    selected_raw_probabilities.append(
        float(
            raw_prob_matrix[
                row_index,
                class_index,
            ]
        )
    )

    selected_normalized_probabilities.append(
        float(
            prob_matrix[
                row_index,
                class_index,
            ]
        )
    )


df[
    "selected_probability_raw"
] = selected_raw_probabilities

df[
    "selected_probability"
] = selected_normalized_probabilities


# ============================================================
# API confidence is NOT assumed to equal probability.
# ============================================================

df["api_confidence"] = (
    df["confidence"]
    .astype(float)
)


max_confidence_probability_difference = float(
    np.max(
        np.abs(
            df["api_confidence"]
            - df[
                "selected_probability_raw"
            ]
        )
    )
)


# ============================================================
# Overall probability diagnostic
# ============================================================

mean_selected_probability = float(
    df[
        "selected_probability"
    ].mean()
)


probability_accuracy_gap = float(
    mean_selected_probability
    - accuracy
)


# ============================================================
# Fixed-bin Expected Calibration Error
#
# Calibration variable:
# normalized probability assigned to selected class
#
# Question:
# "When the selected class gets probability p,
#  how often is it actually correct?"
# ============================================================

bin_edges = np.linspace(
    0.0,
    1.0,
    NUM_ECE_BINS + 1,
)


# np.digitize with internal edges gives bin IDs:
# 0 ... NUM_ECE_BINS - 1
bin_ids = np.digitize(
    df[
        "selected_probability"
    ].to_numpy(),
    bin_edges[1:-1],
    right=False,
)


reliability_rows = []

ece = 0.0
mce = 0.0


for bin_index in range(
    NUM_ECE_BINS
):

    mask = (
        bin_ids == bin_index
    )

    count = int(
        mask.sum()
    )

    lower = float(
        bin_edges[
            bin_index
        ]
    )

    upper = float(
        bin_edges[
            bin_index + 1
        ]
    )


    if count == 0:

        reliability_rows.append(
            {
                "bin":
                    bin_index + 1,

                "lower":
                    lower,

                "upper":
                    upper,

                "count":
                    0,

                "mean_probability":
                    np.nan,

                "accuracy":
                    np.nan,

                "absolute_gap":
                    np.nan,
            }
        )

        continue


    mean_probability = float(
        df.loc[
            mask,
            "selected_probability",
        ].mean()
    )

    bin_accuracy = float(
        df.loc[
            mask,
            "correct",
        ].mean()
    )

    gap = abs(
        mean_probability
        - bin_accuracy
    )


    ece += (
        count / len(df)
    ) * gap

    mce = max(
        mce,
        gap,
    )


    reliability_rows.append(
        {
            "bin":
                bin_index + 1,

            "lower":
                lower,

            "upper":
                upper,

            "count":
                count,

            "mean_probability":
                mean_probability,

            "accuracy":
                bin_accuracy,

            "absolute_gap":
                gap,
        }
    )


reliability_df = pd.DataFrame(
    reliability_rows
)


# ============================================================
# One-hot ground truth
# ============================================================

y_true_one_hot = np.zeros(
    (
        len(df),
        len(LABELS),
    ),
    dtype=float,
)


for row_index, label in enumerate(
    df["v2_label"]
):

    y_true_one_hot[
        row_index,
        label_to_index[
            label
        ],
    ] = 1.0


# ============================================================
# Brier scores
# ============================================================

# Top-label Brier:
#
# Treat the selected class as a binary prediction:
#
#   selected probability vs whether selected class was correct.
#
top_label_brier = float(
    np.mean(
        (
            df[
                "selected_probability"
            ].to_numpy()
            - df[
                "correct"
            ].to_numpy()
        ) ** 2
    )
)


# Multiclass Brier:
#
# Compare entire 9-class probability vector with one-hot truth.
#
multiclass_brier = float(
    np.mean(
        np.sum(
            (
                prob_matrix
                - y_true_one_hot
            ) ** 2,
            axis=1,
        )
    )
)


# ============================================================
# Multiclass log loss
# ============================================================

y_true_indices = np.array(
    [
        label_to_index[
            label
        ]
        for label
        in df["v2_label"]
    ],
    dtype=int,
)


multiclass_log_loss = float(
    log_loss(
        y_true_indices,
        prob_matrix,
        labels=list(
            range(
                len(LABELS)
            )
        ),
    )
)


# ============================================================
# API confidence analysis
#
# We treat confidence as a ranking / risk signal,
# NOT automatically as a calibrated probability.
#
# Question:
#
# Does higher API confidence correspond to higher accuracy?
# ============================================================

confidence_df = df[
    [
        "id",
        "api_confidence",
        "correct",
    ]
].copy()


# qcut creates approximately equal-sized groups.
#
# duplicates="drop" protects against many identical
# confidence values.
confidence_df[
    "confidence_quantile"
] = pd.qcut(
    confidence_df[
        "api_confidence"
    ],
    q=NUM_CONFIDENCE_QUANTILES,
    labels=False,
    duplicates="drop",
)


confidence_quantile_rows = []


for quantile in sorted(
    confidence_df[
        "confidence_quantile"
    ].dropna().unique()
):

    subset = confidence_df[
        confidence_df[
            "confidence_quantile"
        ]
        == quantile
    ]

    count = len(
        subset
    )

    mean_confidence = float(
        subset[
            "api_confidence"
        ].mean()
    )

    minimum_confidence = float(
        subset[
            "api_confidence"
        ].min()
    )

    maximum_confidence = float(
        subset[
            "api_confidence"
        ].max()
    )

    quantile_accuracy = float(
        subset[
            "correct"
        ].mean()
    )

    error_rate = float(
        1.0
        - quantile_accuracy
    )


    confidence_quantile_rows.append(
        {
            "quantile":
                int(
                    quantile
                )
                + 1,

            "count":
                int(
                    count
                ),

            "min_confidence":
                minimum_confidence,

            "max_confidence":
                maximum_confidence,

            "mean_confidence":
                mean_confidence,

            "accuracy":
                quantile_accuracy,

            "error_rate":
                error_rate,
        }
    )


confidence_quantile_df = pd.DataFrame(
    confidence_quantile_rows
)


# ============================================================
# Human-review simulation
#
# Sort by Jev API confidence ascending.
#
# Simulate:
# "Send the lowest-confidence X% to human review."
#
# Measure:
#
# - how many requests need review?
# - how many model errors are contained there?
# - what % of total errors would human review catch?
# - what is the error rate among reviewed cases?
# ============================================================

ranked_by_confidence = (
    df.sort_values(
        "api_confidence",
        ascending=True,
    )
    .reset_index(
        drop=True
    )
)


review_rows = []


for review_fraction in REVIEW_FRACTIONS:

    review_count = int(
        math.ceil(
            len(df)
            * review_fraction
        )
    )

    reviewed = (
        ranked_by_confidence
        .head(
            review_count
        )
    )


    errors_caught = int(
        (
            reviewed[
                "correct"
            ]
            == 0
        ).sum()
    )


    error_capture_rate = (
        errors_caught
        / total_errors
        if total_errors > 0
        else 0.0
    )


    reviewed_error_rate = float(
        1.0
        - reviewed[
            "correct"
        ].mean()
    )


    review_rows.append(
        {
            "review_fraction":
                float(
                    review_fraction
                ),

            "review_percentage":
                float(
                    review_fraction
                    * 100
                ),

            "reviewed_cases":
                int(
                    review_count
                ),

            "total_errors":
                int(
                    total_errors
                ),

            "errors_caught":
                int(
                    errors_caught
                ),

            "error_capture_rate":
                float(
                    error_capture_rate
                ),

            "reviewed_error_rate":
                reviewed_error_rate,

            "max_confidence_reviewed":
                float(
                    reviewed[
                        "api_confidence"
                    ].max()
                ),
        }
    )


review_df = pd.DataFrame(
    review_rows
)


# ============================================================
# Print report
# ============================================================

print("=" * 78)
print("JEV CALIBRATION AND CONFIDENCE ANALYSIS")
print("=" * 78)

print(
    f"Examples                         : "
    f"{len(df):,}"
)

print(
    f"Correct                          : "
    f"{int(df['correct'].sum()):,}"
)

print(
    f"Errors                           : "
    f"{total_errors:,}"
)

print(
    f"Accuracy                         : "
    f"{accuracy:.4f}"
)


print()
print("Probability-vector diagnostics")
print("-" * 78)

print(
    f"Maximum raw probability-sum error     : "
    f"{max_probability_sum_error:.8f}"
)

print(
    f"Maximum normalized probability error  : "
    f"{max_normalized_sum_error:.8f}"
)

print(
    f"Max API confidence/raw-prob difference: "
    f"{max_confidence_probability_difference:.8f}"
)


print()
print("Probability calibration")
print("-" * 78)

print(
    f"Mean selected-class probability : "
    f"{mean_selected_probability:.4f}"
)

print(
    f"Accuracy                        : "
    f"{accuracy:.4f}"
)

print(
    f"Probability - accuracy gap      : "
    f"{probability_accuracy_gap:+.4f}"
)

print(
    f"ECE                             : "
    f"{ece:.4f}"
)

print(
    f"MCE                             : "
    f"{mce:.4f}"
)

print(
    f"Top-label Brier score           : "
    f"{top_label_brier:.4f}"
)

print(
    f"Multiclass Brier score          : "
    f"{multiclass_brier:.4f}"
)

print(
    f"Multiclass log loss             : "
    f"{multiclass_log_loss:.4f}"
)


print()
print("Reliability bins")
print("-" * 78)

print(
    reliability_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


print()
print("API confidence quantiles")
print("-" * 78)

print(
    confidence_quantile_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


print()
print("Human-review simulation")
print("-" * 78)

print(
    review_df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}"
        ),
    )
)


# ============================================================
# Save CSV outputs
# ============================================================

reliability_file = (
    OUTPUT_DIR
    / "jev_reliability_bins.csv"
)

confidence_quantile_file = (
    OUTPUT_DIR
    / "jev_confidence_quantiles.csv"
)

review_file = (
    OUTPUT_DIR
    / "jev_confidence_review.csv"
)

case_file = (
    OUTPUT_DIR
    / "jev_calibration_cases.csv"
)

summary_file = (
    OUTPUT_DIR
    / "jev_calibration_summary.json"
)


reliability_df.to_csv(
    reliability_file,
    index=False,
)

confidence_quantile_df.to_csv(
    confidence_quantile_file,
    index=False,
)

review_df.to_csv(
    review_file,
    index=False,
)


df[
    [
        "id",
        "prediction",
        "v2_label",
        "correct",
        "selected_probability_raw",
        "selected_probability",
        "api_confidence",
    ]
].to_csv(
    case_file,
    index=False,
)


# ============================================================
# Save JSON summary
# ============================================================

summary = {
    "examples":
        int(
            len(df)
        ),

    "correct":
        int(
            df["correct"].sum()
        ),

    "errors":
        total_errors,

    "accuracy":
        accuracy,

    "probability_diagnostics": {
        "max_raw_probability_sum_error":
            max_probability_sum_error,

        "max_normalized_probability_sum_error":
            max_normalized_sum_error,

        "max_api_confidence_raw_probability_difference":
            max_confidence_probability_difference,
    },

    "probability_calibration": {
        "mean_selected_probability":
            mean_selected_probability,

        "probability_minus_accuracy":
            probability_accuracy_gap,

        "ece":
            float(
                ece
            ),

        "mce":
            float(
                mce
            ),

        "top_label_brier":
            top_label_brier,

        "multiclass_brier":
            multiclass_brier,

        "multiclass_log_loss":
            multiclass_log_loss,

        "ece_bins":
            NUM_ECE_BINS,
    },

    "api_confidence": {
        "mean":
            float(
                df[
                    "api_confidence"
                ].mean()
            ),

        "min":
            float(
                df[
                    "api_confidence"
                ].min()
            ),

        "max":
            float(
                df[
                    "api_confidence"
                ].max()
            ),

        "quantile_groups":
            int(
                len(
                    confidence_quantile_df
                )
            ),
    },

    "review_simulation":
        review_df.to_dict(
            orient="records"
        ),
}


with summary_file.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# Figure 1: probability reliability diagram
# ============================================================

plot_df = reliability_df.dropna(
    subset=[
        "mean_probability",
        "accuracy",
    ]
)


plt.figure(
    figsize=(7, 7)
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)

plt.plot(
    plot_df[
        "mean_probability"
    ],
    plot_df[
        "accuracy"
    ],
    marker="o",
    label="Jev",
)

plt.xlabel(
    "Mean selected-class probability"
)

plt.ylabel(
    "Observed accuracy"
)

plt.title(
    "Jev Probability Reliability Diagram"
)

plt.xlim(
    0,
    1,
)

plt.ylim(
    0,
    1,
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()


reliability_figure = (
    FIGURES_DIR
    / "jev_reliability_diagram.png"
)

plt.savefig(
    reliability_figure,
    dpi=180,
)

plt.close()


# ============================================================
# Figure 2: API confidence vs observed accuracy
#
# This does NOT interpret confidence as probability.
# It tests whether higher confidence corresponds to
# higher empirical accuracy.
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    confidence_quantile_df[
        "quantile"
    ],
    confidence_quantile_df[
        "accuracy"
    ],
    marker="o",
)

plt.xlabel(
    "API confidence quantile "
    "(1 = lowest confidence)"
)

plt.ylabel(
    "Observed accuracy"
)

plt.title(
    "Jev Accuracy by API Confidence Quantile"
)

plt.ylim(
    0,
    1,
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()


confidence_figure = (
    FIGURES_DIR
    / "jev_confidence_accuracy.png"
)

plt.savefig(
    confidence_figure,
    dpi=180,
)

plt.close()


# ============================================================
# Figure 3: review fraction vs errors captured
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    review_df[
        "review_percentage"
    ],
    review_df[
        "error_capture_rate"
    ]
    * 100,
    marker="o",
)

plt.xlabel(
    "Requests sent to human review (%)"
)

plt.ylabel(
    "Model errors captured (%)"
)

plt.title(
    "Jev Low-Confidence Review Efficiency"
)

plt.ylim(
    0,
    100,
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()


review_figure = (
    FIGURES_DIR
    / "jev_confidence_review.png"
)

plt.savefig(
    review_figure,
    dpi=180,
)

plt.close()


# ============================================================
# Final output list
# ============================================================

print()
print("Files created")
print("-" * 78)

for path in [
    reliability_file,
    confidence_quantile_file,
    review_file,
    case_file,
    summary_file,
    reliability_figure,
    confidence_figure,
    review_figure,
]:
    print(
        f"  {path.relative_to(ROOT)}"
    )