"""Sentence embeddings + Logistic Regression baseline for V1."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = ROOT / "data" / "prepared" / "train_ground_truth.csv"
TEST_INPUT_FILE = ROOT / "data" / "prepared" / "test_600_inputs.csv"

CACHE_DIR = ROOT / "data" / "cache"

PREDICTIONS_DIR = ROOT / "results" / "predictions"
TIMINGS_DIR = ROOT / "results" / "timings"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
TIMINGS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_EMBEDDINGS_FILE = CACHE_DIR / "minilm_train_embeddings.npy"
TEST_EMBEDDINGS_FILE = CACHE_DIR / "minilm_test600_embeddings.npy"

PREDICTIONS_FILE = PREDICTIONS_DIR / "embedding_lr_v1.csv"
TIMINGS_FILE = TIMINGS_DIR / "embedding_lr_v1.json"


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42
BATCH_SIZE = 64


# ============================================================
# Load data
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_INPUT_FILE)

X_train_text = train_df["text"].astype(str).tolist()
y_train = train_df["v1_label"]

X_test_text = test_df["text"].astype(str).tolist()


# ============================================================
# Load embedding model
# ============================================================

print("=" * 70)
print("EMBEDDING + LOGISTIC REGRESSION — V1")
print("=" * 70)

print(f"Embedding model   : {MODEL_NAME}")
print(f"Training examples : {len(train_df):,}")
print(f"Test examples     : {len(test_df):,}")
print(f"V1 classes        : {y_train.nunique()}")

print()
print("Loading embedding model...")

embedding_model = SentenceTransformer(MODEL_NAME)


# ============================================================
# Build / load training embeddings
# ============================================================

if TRAIN_EMBEDDINGS_FILE.exists():
    print()
    print("Loading cached training embeddings...")

    X_train = np.load(TRAIN_EMBEDDINGS_FILE)

else:
    print()
    print("Creating training embeddings...")

    start = time.perf_counter()

    X_train = embedding_model.encode(
        X_train_text,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding_train_seconds = time.perf_counter() - start

    np.save(
        TRAIN_EMBEDDINGS_FILE,
        X_train,
    )

    print(
        f"Training embedding time: "
        f"{embedding_train_seconds:.3f} s"
    )


# ============================================================
# Build / load test embeddings
# ============================================================

if TEST_EMBEDDINGS_FILE.exists():
    print()
    print("Loading cached test embeddings...")

    X_test = np.load(TEST_EMBEDDINGS_FILE)

else:
    print()
    print("Creating test embeddings...")

    start = time.perf_counter()

    X_test = embedding_model.encode(
        X_test_text,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding_test_seconds = time.perf_counter() - start

    np.save(
        TEST_EMBEDDINGS_FILE,
        X_test,
    )

    print(
        f"Test embedding time: "
        f"{embedding_test_seconds:.3f} s"
    )


# ============================================================
# Validate embedding shapes
# ============================================================

assert len(X_train) == len(train_df)
assert len(X_test) == len(test_df)

embedding_dimensions = X_train.shape[1]

print()
print(
    f"Embedding dimension: "
    f"{embedding_dimensions}"
)


# ============================================================
# Train Logistic Regression
# ============================================================

classifier = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_SEED,
)

print()
print("Training Logistic Regression...")

train_start = time.perf_counter()

classifier.fit(
    X_train,
    y_train,
)

classifier_train_seconds = (
    time.perf_counter() - train_start
)

print(
    f"Classifier training time: "
    f"{classifier_train_seconds:.3f} s"
)


# ============================================================
# Batch prediction
# ============================================================

batch_start = time.perf_counter()

predictions = classifier.predict(
    X_test
)

batch_prediction_seconds = (
    time.perf_counter() - batch_start
)

print(
    f"Classifier batch prediction: "
    f"{batch_prediction_seconds:.3f} s "
    f"for {len(test_df):,} examples"
)


# ============================================================
# End-to-end latency measurement
#
# This includes:
#   text -> embedding -> classifier
#
# This is what matters when comparing against TF-IDF,
# SLM, and Jev.
# ============================================================

print()
print("Measuring end-to-end per-example latency...")

latencies_ms = []

for text in X_test_text:

    start = time.perf_counter()

    embedding = embedding_model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    classifier.predict(
        embedding
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    latencies_ms.append(
        elapsed_ms
    )


latencies = np.array(
    latencies_ms
)

mean_latency_ms = float(
    np.mean(latencies)
)

p50_latency_ms = float(
    np.percentile(latencies, 50)
)

p95_latency_ms = float(
    np.percentile(latencies, 95)
)


# ============================================================
# Save predictions
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
# Save timing metadata
# ============================================================

timing_data = {
    "model": "embedding_lr",
    "task": "v1",

    "embedding_model": MODEL_NAME,

    "training_examples": int(
        len(train_df)
    ),

    "test_examples": int(
        len(test_df)
    ),

    "classes": int(
        y_train.nunique()
    ),

    "embedding_dimensions": int(
        embedding_dimensions
    ),

    "classifier_training_seconds": float(
        classifier_train_seconds
    ),

    "classifier_batch_prediction_seconds": float(
        batch_prediction_seconds
    ),

    "latency_ms": {
        "mean": mean_latency_ms,
        "p50": p50_latency_ms,
        "p95": p95_latency_ms,
    },

    "configuration": {
        "batch_size": BATCH_SIZE,
        "normalize_embeddings": True,
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
    f"Embedding dimension : "
    f"{embedding_dimensions}"
)

print(
    f"Mean latency        : "
    f"{mean_latency_ms:.3f} ms"
)

print(
    f"p50 latency         : "
    f"{p50_latency_ms:.3f} ms"
)

print(
    f"p95 latency         : "
    f"{p95_latency_ms:.3f} ms"
)

print()
print("Files created:")

print(
    "  results/predictions/"
    "embedding_lr_v1.csv"
)

print(
    "  results/timings/"
    "embedding_lr_v1.json"
)

print()
print("Cached embeddings:")

print(
    "  data/cache/"
    "minilm_train_embeddings.npy"
)

print(
    "  data/cache/"
    "minilm_test600_embeddings.npy"
)