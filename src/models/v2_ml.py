"""V2 adaptation experiments for supervised ML baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PREPARED_DIR = ROOT / "data" / "prepared"

CLEAN_HISTORY_FILE = (
    PREPARED_DIR / "clean_v1_reusable.csv"
)

TEST_INPUT_FILE = (
    PREPARED_DIR / "test_600_inputs.csv"
)

ADAPTATION_DIR = (
    PREPARED_DIR / "adaptation"
)

PREDICTIONS_DIR = (
    ROOT / "results" / "predictions"
)

TIMINGS_DIR = (
    ROOT / "results" / "timings"
)

CACHE_DIR = (
    ROOT / "data" / "cache"
)

PREDICTIONS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TIMINGS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

BATCH_SIZE = 64


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "tfidf",
            "embedding",
        ],
    )

    parser.add_argument(
        "--budget",
        required=True,
        type=int,
        choices=[
            20,
            100,
        ],
    )

    parser.add_argument(
        "--draw",
        required=True,
        type=int,
        choices=range(1, 6),
    )

    return parser.parse_args()


# ============================================================
# Load V2 training data
# ============================================================

def load_training_data(
    budget: int,
    draw: int,
) -> pd.DataFrame:

    clean_df = pd.read_csv(
        CLEAN_HISTORY_FILE
    )

    adaptation_file = (
        ADAPTATION_DIR
        / f"draw_{draw:02d}"
        / f"v2_{budget}.csv"
    )

    adaptation_df = pd.read_csv(
        adaptation_file
    )

    clean_part = clean_df[
        [
            "id",
            "text",
            "v2_label",
        ]
    ].copy()

    adaptation_part = adaptation_df[
        [
            "id",
            "text",
            "v2_label",
        ]
    ].copy()

    combined = pd.concat(
        [
            clean_part,
            adaptation_part,
        ],
        ignore_index=True,
    )

    if combined["id"].duplicated().any():
        raise ValueError(
            "Duplicate training IDs found."
        )

    return combined


# ============================================================
# TF-IDF experiment
# ============================================================

def run_tfidf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):

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
                    class_weight="balanced",
                ),
            ),
        ]
    )

    start = time.perf_counter()

    model.fit(
        train_df["text"],
        train_df["v2_label"],
    )

    training_seconds = (
        time.perf_counter() - start
    )

    start = time.perf_counter()

    predictions = model.predict(
        test_df["text"]
    )

    batch_seconds = (
        time.perf_counter() - start
    )


    # End-to-end single request latency
    latencies_ms = []

    for text in test_df["text"]:

        start = time.perf_counter()

        model.predict([text])

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000

        latencies_ms.append(
            elapsed
        )

    latencies = np.array(
        latencies_ms
    )

    metadata = {
        "training_seconds":
            float(training_seconds),

        "batch_prediction_seconds":
            float(batch_seconds),

        "p50_latency_ms":
            float(
                np.percentile(
                    latencies,
                    50,
                )
            ),

        "p95_latency_ms":
            float(
                np.percentile(
                    latencies,
                    95,
                )
            ),
    }

    return predictions, metadata


# ============================================================
# Embedding experiment
# ============================================================

def run_embedding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    budget: int,
    draw: int,
):

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    train_cache = (
        CACHE_DIR
        / (
            f"minilm_v2_"
            f"b{budget}_"
            f"d{draw:02d}_train.npy"
        )
    )

    # ----------------------------------------
    # Training embeddings
    # ----------------------------------------

    if train_cache.exists():

        X_train = np.load(
            train_cache
        )

        embedding_training_seconds = 0.0

    else:

        start = time.perf_counter()

        X_train = embedding_model.encode(
            train_df["text"].tolist(),
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embedding_training_seconds = (
            time.perf_counter()
            - start
        )

        np.save(
            train_cache,
            X_train,
        )


    # ----------------------------------------
    # Reuse frozen test embeddings
    # ----------------------------------------

    test_cache = (
        CACHE_DIR
        / "minilm_test600_embeddings.npy"
    )

    if not test_cache.exists():
        raise FileNotFoundError(
            "Expected cached test embeddings "
            "from the V1 embedding experiment."
        )

    X_test = np.load(
        test_cache
    )


    # ----------------------------------------
    # Train classifier
    # ----------------------------------------

    classifier = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )

    start = time.perf_counter()

    classifier.fit(
        X_train,
        train_df["v2_label"],
    )

    classifier_training_seconds = (
        time.perf_counter()
        - start
    )


    # ----------------------------------------
    # Batch prediction
    # ----------------------------------------

    start = time.perf_counter()

    predictions = classifier.predict(
        X_test
    )

    batch_seconds = (
        time.perf_counter()
        - start
    )


    # ----------------------------------------
    # End-to-end latency
    #
    # text -> embedding -> classifier
    # ----------------------------------------

    latencies_ms = []

    for text in test_df["text"]:

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

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000

        latencies_ms.append(
            elapsed
        )

    latencies = np.array(
        latencies_ms
    )


    metadata = {
        "embedding_training_seconds":
            float(
                embedding_training_seconds
            ),

        "classifier_training_seconds":
            float(
                classifier_training_seconds
            ),

        "batch_prediction_seconds":
            float(
                batch_seconds
            ),

        "p50_latency_ms":
            float(
                np.percentile(
                    latencies,
                    50,
                )
            ),

        "p95_latency_ms":
            float(
                np.percentile(
                    latencies,
                    95,
                )
            ),
    }

    return predictions, metadata


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    train_df = load_training_data(
        budget=args.budget,
        draw=args.draw,
    )

    test_df = pd.read_csv(
        TEST_INPUT_FILE
    )


    print("=" * 70)

    print(
        f"V2 ADAPTATION — "
        f"{args.model.upper()}"
    )

    print("=" * 70)

    print(
        f"Budget             : "
        f"{args.budget} labels/class"
    )

    print(
        f"Draw               : "
        f"{args.draw}"
    )

    print(
        f"Training examples  : "
        f"{len(train_df):,}"
    )

    print(
        f"V2 classes         : "
        f"{train_df['v2_label'].nunique()}"
    )


    if args.model == "tfidf":

        predictions, metadata = (
            run_tfidf(
                train_df,
                test_df,
            )
        )

    else:

        predictions, metadata = (
            run_embedding(
                train_df,
                test_df,
                args.budget,
                args.draw,
            )
        )


    # ========================================================
    # Save predictions
    # ========================================================

    experiment_name = (
        f"{args.model}_lr_"
        f"v2_b{args.budget}_"
        f"d{args.draw:02d}"
    )

    predictions_file = (
        PREDICTIONS_DIR
        / f"{experiment_name}.csv"
    )

    pd.DataFrame(
        {
            "id": test_df["id"],
            "prediction": predictions,
        }
    ).to_csv(
        predictions_file,
        index=False,
    )


    # ========================================================
    # Save timings
    # ========================================================

    metadata.update(
        {
            "model":
                args.model,

            "task":
                "v2",

            "budget":
                args.budget,

            "draw":
                args.draw,

            "training_examples":
                len(train_df),

            "new_labels":
                args.budget * 6,
        }
    )

    timing_file = (
        TIMINGS_DIR
        / f"{experiment_name}.json"
    )

    with timing_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


    print()
    print("Complete.")

    print(
        f"Predictions: "
        f"{predictions_file.relative_to(ROOT)}"
    )

    print(
        f"Timings: "
        f"{timing_file.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()