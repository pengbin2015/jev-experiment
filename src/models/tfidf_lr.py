"""TF-IDF + Logistic Regression baseline for V1 classification."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data" / "prepared" / "train_ground_truth.csv"
TEST_INPUT_FILE = ROOT / "data" / "prepared" / "test_600_inputs.csv"

PREDICTIONS_DIR = ROOT / "results" / "predictions"
TIMINGS_DIR = ROOT / "results" / "timings"

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
TIMINGS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = PREDICTIONS_DIR / "tfidf_lr_v1.csv"
TIMINGS_FILE = TIMINGS_DIR / "tfidf_lr_v1.json"


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42


# ============================================================
# Load data
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_INPUT_FILE)


required_train_columns = {
    "id",
    "text",
    "v1_label",
}

required_test_columns = {
    "id",
    "text",
}

missing_train = required_train_columns - set(train_df.columns)
missing_test = required_test_columns - set(test_df.columns)

if missing_train:
    raise ValueError(
        f"Training data missing columns: {sorted(missing_train)}"
    )

if missing_test:
    raise ValueError(
        f"Test input missing columns: {sorted(missing_test)}"
    )


# ============================================================
# Prepare X/y
# ============================================================

X_train = train_df["text"]
y_train = train_df["v1_label"]

X_test = test_df["text"]


# ============================================================
# Model
#
# TF-IDF:
#   converts text into sparse numerical features
#
# Logistic Regression:
#   supervised multiclass classifier
# ============================================================

model = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_SEED,
            ),
        ),
    ]
)


# ============================================================
# Train
# ============================================================

print("=" * 70)
print("TF-IDF + LOGISTIC REGRESSION — V1")
print("=" * 70)

print(f"Training examples : {len(train_df):,}")
print(f"Test examples     : {len(test_df):,}")
print(f"V1 classes        : {y_train.nunique()}")

print()
print("Training model...")

train_start = time.perf_counter()

model.fit(
    X_train,
    y_train,
)

train_seconds = time.perf_counter() - train_start

print(f"Training time     : {train_seconds:.3f} s")


# ============================================================
# Batch prediction
#
# Use normal batch prediction to produce the benchmark outputs.
# ============================================================

batch_start = time.perf_counter()

predictions = model.predict(X_test)

batch_seconds = time.perf_counter() - batch_start

print(
    f"Batch prediction  : {batch_seconds:.3f} s "
    f"for {len(test_df):,} examples"
)


# ============================================================
# Per-example end-to-end latency
#
# This deliberately includes:
#
#   text -> TF-IDF transform -> classifier prediction
#
# so it represents model-side latency for one request.
#
# We repeat prediction one example at a time only for timing.
# These predictions are NOT used as separate benchmark outputs.
# ============================================================

print()
print("Measuring per-example latency...")

latencies_ms = []

for text in X_test:
    start = time.perf_counter()

    model.predict([text])

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    latencies_ms.append(elapsed_ms)


latencies = np.array(latencies_ms)

p50_latency_ms = float(
    np.percentile(latencies, 50)
)

p95_latency_ms = float(
    np.percentile(latencies, 95)
)

mean_latency_ms = float(
    np.mean(latencies)
)


# ============================================================
# Save predictions
#
# IMPORTANT:
# No ground-truth labels are included.
# ============================================================

prediction_df = pd.DataFrame(
    {
        "id": test_df["id"],
        "prediction": predictions,
    }
)

prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


# ============================================================
# Model information
# ============================================================

tfidf = model.named_steps["tfidf"]

vocabulary_size = len(
    tfidf.vocabulary_
)


# ============================================================
# Save timing / configuration metadata
# ============================================================

timing_data = {
    "model": "tfidf_lr",
    "task": "v1",
    "training_examples": int(len(train_df)),
    "test_examples": int(len(test_df)),
    "classes": int(y_train.nunique()),
    "vocabulary_size": int(vocabulary_size),

    "training_seconds": float(train_seconds),
    "batch_prediction_seconds": float(batch_seconds),

    "latency_ms": {
        "mean": mean_latency_ms,
        "p50": p50_latency_ms,
        "p95": p95_latency_ms,
    },

    "configuration": {
        "tfidf": {
            "lowercase": True,
            "ngram_range": [1, 2],
            "min_df": 2,
            "sublinear_tf": True,
        },
        "logistic_regression": {
            "max_iter": 2000,
            "random_state": RANDOM_SEED,
        },
    },
}

with TIMINGS_FILE.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        timing_data,
        f,
        indent=2,
    )


# ============================================================
# Report
# ============================================================

print()
print("MODEL COMPLETE")
print("-" * 70)

print(
    f"Vocabulary size   : {vocabulary_size:,}"
)

print(
    f"Mean latency      : {mean_latency_ms:.3f} ms"
)

print(
    f"p50 latency       : {p50_latency_ms:.3f} ms"
)

print(
    f"p95 latency       : {p95_latency_ms:.3f} ms"
)

print()
print("Files created:")
print(
    "  results/predictions/tfidf_lr_v1.csv"
)
print(
    "  results/timings/tfidf_lr_v1.json"
)