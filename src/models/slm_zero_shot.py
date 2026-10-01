"""Zero-shot SLM baseline for V2 classification using Ollama."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_INPUT_FILE = (
    ROOT
    / "data"
    / "prepared"
    / "test_600_inputs.csv"
)

V2_CLASSES_FILE = (
    ROOT
    / "config"
    / "v2_classes.yaml"
)

PREDICTIONS_DIR = (
    ROOT
    / "results"
    / "predictions"
)

TIMINGS_DIR = (
    ROOT
    / "results"
    / "timings"
)

RAW_DIR = (
    ROOT
    / "results"
    / "raw"
)

PREDICTIONS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TIMINGS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = (
    "qwen3:4b-instruct-2507-q4_K_M"
)

OLLAMA_URL = (
    "http://localhost:11434/api/chat"
)

RANDOM_SEED = 42


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run zero-shot V2 classification "
            "using a local Ollama SLM."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional limit for a smoke test. "
            "Do not use for final evaluation."
        ),
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Ignore existing partial results "
            "and start again."
        ),
    )

    return parser.parse_args()


# ============================================================
# Load V2 business taxonomy
# ============================================================

def load_v2_classes() -> dict[str, str]:

    with V2_CLASSES_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    classes = config["classes"]

    result = {}

    for label, item in classes.items():
        result[label] = (
            item["description"]
            .strip()
        )

    if len(result) != 9:
        raise ValueError(
            f"Expected 9 V2 classes, "
            f"found {len(result)}."
        )

    return result


# ============================================================
# Frozen prompt
# ============================================================

def build_system_prompt(
    classes: dict[str, str],
) -> str:

    taxonomy_text = "\n".join(
        (
            f"- {label}: "
            f"{description}"
        )
        for label, description
        in classes.items()
    )

    return f"""You classify online banking customer requests.

Choose exactly one class from the business taxonomy below.

Use only the customer's message and the class definitions.
Choose the single best matching class.
Do not invent new classes.
Do not explain your answer.

Return exactly one class name and nothing else.

Business taxonomy:

{taxonomy_text}
"""


# ============================================================
# Parse model output
# ============================================================

def parse_prediction(
    raw_output: str,
    valid_labels: set[str],
) -> str:

    cleaned = (
        raw_output
        .strip()
        .upper()
    )

    # Perfect response.
    if cleaned in valid_labels:
        return cleaned

    # Strip common surrounding punctuation.
    simplified = re.sub(
        r"[^A-Z_]",
        " ",
        cleaned,
    )

    found = []

    for label in valid_labels:
        pattern = (
            r"\b"
            + re.escape(label)
            + r"\b"
        )

        if re.search(
            pattern,
            simplified,
        ):
            found.append(label)

    # Accept only when exactly one valid label
    # appears in the response.
    if len(found) == 1:
        return found[0]

    return "INVALID"


# ============================================================
# Call Ollama
# ============================================================

def call_model(
    model: str,
    system_prompt: str,
    customer_text: str,
) -> tuple[str, float]:

    payload = {
        "model": model,
        "stream": False,

        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Customer message:\n"
                    f"{customer_text}\n\n"
                    "Class:"
                ),
            },
        ],

        "options": {
            "temperature": 0,
            "seed": RANDOM_SEED,
            "num_predict": 16,
        },
    }

    start = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300,
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    response.raise_for_status()

    data = response.json()

    raw_output = (
        data["message"]["content"]
        .strip()
    )

    return raw_output, latency_ms


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    test_df = pd.read_csv(
        TEST_INPUT_FILE
    )

    if args.limit is not None:
        test_df = test_df.head(
            args.limit
        )

    classes = load_v2_classes()

    valid_labels = set(
        classes.keys()
    )

    system_prompt = (
        build_system_prompt(
            classes
        )
    )


    # --------------------------------------------------------
    # Output filenames
    # --------------------------------------------------------

    experiment_name = (
        "slm_qwen3_4b_v2"
    )

    raw_file = (
        RAW_DIR
        / f"{experiment_name}_raw.csv"
    )

    predictions_file = (
        PREDICTIONS_DIR
        / f"{experiment_name}.csv"
    )

    timings_file = (
        TIMINGS_DIR
        / f"{experiment_name}.json"
    )


    # --------------------------------------------------------
    # Existing partial results
    # --------------------------------------------------------

    if (
        raw_file.exists()
        and not args.restart
    ):
        results_df = pd.read_csv(
            raw_file
        )

        completed_ids = set(
            results_df["id"]
        )

        print(
            f"Resuming with "
            f"{len(completed_ids):,} "
            f"existing predictions."
        )

    else:
        results_df = pd.DataFrame(
            columns=[
                "id",
                "prediction",
                "latency_ms",
                "raw_output",
            ]
        )

        completed_ids = set()


    # --------------------------------------------------------
    # Report configuration
    # --------------------------------------------------------

    print("=" * 70)
    print("ZERO-SHOT SLM — V2")
    print("=" * 70)

    print(
        f"Model          : "
        f"{args.model}"
    )

    print(
        f"Test examples  : "
        f"{len(test_df):,}"
    )

    print(
        f"V2 classes     : "
        f"{len(classes)}"
    )

    print(
        "New V2 labels  : 0"
    )

    print()


    # --------------------------------------------------------
    # Warm-up request
    #
    # Discarded; avoids including initial model loading
    # time in the latency benchmark.
    # --------------------------------------------------------

    if not completed_ids:

        print("Warming up model...")

        call_model(
            model=args.model,
            system_prompt=system_prompt,
            customer_text=(
                "I need help with my bank account."
            ),
        )


    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    total = len(test_df)

    for index, row in test_df.iterrows():

        example_id = row["id"]

        if example_id in completed_ids:
            continue

        raw_output, latency_ms = (
            call_model(
                model=args.model,
                system_prompt=system_prompt,
                customer_text=str(
                    row["text"]
                ),
            )
        )

        prediction = (
            parse_prediction(
                raw_output,
                valid_labels,
            )
        )

        new_row = pd.DataFrame(
            [
                {
                    "id": example_id,
                    "prediction": prediction,
                    "latency_ms": latency_ms,
                    "raw_output": raw_output,
                }
            ]
        )

        results_df = pd.concat(
            [
                results_df,
                new_row,
            ],
            ignore_index=True,
        )

        # Save after every request so the run
        # can safely resume after interruption.
        results_df.to_csv(
            raw_file,
            index=False,
        )

        completed = len(
            set(results_df["id"])
            & set(test_df["id"])
        )

        if (
            completed % 25 == 0
            or completed == total
        ):
            print(
                f"Completed "
                f"{completed:,}/{total:,}"
            )


    # --------------------------------------------------------
    # Keep only rows belonging to this requested run
    # --------------------------------------------------------

    final_df = (
        results_df[
            results_df["id"]
            .isin(test_df["id"])
        ]
        .drop_duplicates(
            subset=["id"],
            keep="last",
        )
    )


    # --------------------------------------------------------
    # Save clean prediction file
    #
    # Evaluator sees only id + prediction.
    # --------------------------------------------------------

    prediction_df = final_df[
        [
            "id",
            "prediction",
        ]
    ].copy()

    prediction_df.to_csv(
        predictions_file,
        index=False,
    )


    # --------------------------------------------------------
    # Timing statistics
    # --------------------------------------------------------

    latencies = (
        final_df["latency_ms"]
        .astype(float)
        .to_numpy()
    )

    invalid_count = int(
        (
            final_df["prediction"]
            == "INVALID"
        ).sum()
    )

    metadata = {
        "model": args.model,
        "task": "v2",
        "method": "zero_shot_slm",
        "new_v2_labels": 0,
        "test_examples": int(
            len(final_df)
        ),
        "invalid_outputs": invalid_count,

        "latency_ms": {
            "mean": float(
                np.mean(latencies)
            ),
            "p50": float(
                np.percentile(
                    latencies,
                    50,
                )
            ),
            "p95": float(
                np.percentile(
                    latencies,
                    95,
                )
            ),
        },

        "generation": {
            "temperature": 0,
            "seed": RANDOM_SEED,
            "num_predict": 16,
        },
    }

    with timings_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("MODEL COMPLETE")
    print("-" * 70)

    print(
        f"Predictions     : "
        f"{len(final_df):,}"
    )

    print(
        f"Invalid outputs : "
        f"{invalid_count}"
    )

    print(
        f"Mean latency    : "
        f"{metadata['latency_ms']['mean']:.2f} ms"
    )

    print(
        f"p50 latency     : "
        f"{metadata['latency_ms']['p50']:.2f} ms"
    )

    print(
        f"p95 latency     : "
        f"{metadata['latency_ms']['p95']:.2f} ms"
    )

    print()
    print("Files created:")

    print(
        f"  {predictions_file.relative_to(ROOT)}"
    )

    print(
        f"  {raw_file.relative_to(ROOT)}"
    )

    print(
        f"  {timings_file.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()