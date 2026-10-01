"""Zero-shot Jev baseline for V2 classification."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv


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

RAW_DIR = (
    ROOT
    / "results"
    / "raw"
)

TIMINGS_DIR = (
    ROOT
    / "results"
    / "timings"
)

PREDICTIONS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TIMINGS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# API configuration
# ============================================================

BASE_URL = "https://api.typesafe.ai"

MODELS_URL = (
    f"{BASE_URL}/v1/models"
)

SYSTEMONE_URL = (
    f"{BASE_URL}/v1/systemone"
)


# TypeSafe public pricing as of 2026-09-30:
# $42 / billion input tokens
# = $0.042 / million input tokens
#
# Output tokens are currently free.
INPUT_PRICE_PER_MILLION = 0.042


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Jev zero-shot classification "
            "against the frozen V2 taxonomy."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. "
            "Do not use for final evaluation."
        ),
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Ignore partial results and "
            "restart from the beginning."
        ),
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help=(
            "List models available to the "
            "authenticated TypeSafe account."
        ),
    )

    return parser.parse_args()


# ============================================================
# Load environment
# ============================================================

def get_api_key() -> str:

    load_dotenv(
        ROOT / ".env"
    )

    api_key = os.getenv(
        "TYPESAFE_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "TYPESAFE_API_KEY is not set. "
            "Add it to your .env file."
        )

    return api_key


def get_headers(
    api_key: str,
) -> dict[str, str]:

    return {
        "Authorization": (
            f"Bearer {api_key}"
        ),
        "Content-Type": (
            "application/json"
        ),
    }


# ============================================================
# Discover available models
# ============================================================

def get_available_models(
    headers: dict[str, str],
) -> list[dict]:

    response = requests.get(
        MODELS_URL,
        headers=headers,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    return data["models"]


def choose_model(
    models: list[dict],
) -> str:

    configured_model = os.getenv(
        "TYPESAFE_MODEL"
    )

    available_names = [
        model["name"]
        for model in models
    ]

    if configured_model:

        if (
            configured_model
            not in available_names
        ):
            raise ValueError(
                f"TYPESAFE_MODEL="
                f"{configured_model!r} "
                "is not available.\n"
                f"Available models: "
                f"{available_names}"
            )

        return configured_model


    # Prefer the documented alias if
    # available to this account.
    if "jev-latest" in available_names:
        return "jev-latest"


    if not available_names:
        raise RuntimeError(
            "No TypeSafe models are "
            "available to this account."
        )


    # If the alias is unavailable, use the
    # first model returned by the API and
    # record it in the benchmark metadata.
    return available_names[0]


# ============================================================
# Load frozen V2 business taxonomy
# ============================================================

def load_v2_classes() -> dict[str, str]:

    with V2_CLASSES_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    classes = {}

    for label, item in (
        config["classes"].items()
    ):
        classes[label] = (
            item["description"]
            .strip()
        )

    if len(classes) != 9:
        raise ValueError(
            f"Expected 9 V2 classes; "
            f"found {len(classes)}."
        )

    return classes


# ============================================================
# Jev request
# ============================================================

def classify(
    headers: dict[str, str],
    model: str,
    customer_text: str,
    classes: dict[str, str],
) -> dict:

    payload = {
        "model": model,

        # Same raw customer message that
        # the SLM received.
        "state": customer_text,

        "questions": {
            "business_queue": {
                "type": "choice",

                "instructions": (
                    "Choose the single best "
                    "business queue for this "
                    "online banking customer "
                    "request."
                ),

                # Exact same frozen V2
                # definitions used by Qwen.
                "criteria": classes,
            }
        },
    }


    start = time.perf_counter()

    response = requests.post(
        SYSTEMONE_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000


    response.raise_for_status()

    data = response.json()

    answer = (
        data["answers"]
        ["business_queue"]
    )


    if answer["type"] != "choice":
        raise ValueError(
            "Expected a choice answer, "
            f"received: {answer['type']}"
        )


    return {
        "prediction": (
            answer["choice"]
        ),

        "confidence": float(
            answer["confidence"]
        ),

        "probabilities": (
            answer["probabilities"]
        ),

        "latency_ms": float(
            latency_ms
        ),

        "response_model": (
            data["model"]
        ),

        "input_tokens": int(
            data["usage"]["input_tokens"]
        ),

        "output_tokens": int(
            data["usage"]["output_tokens"]
        ),

        "raw_response": data,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    api_key = get_api_key()

    headers = get_headers(
        api_key
    )

    models = get_available_models(
        headers
    )


    # --------------------------------------------------------
    # Optional model discovery only
    # --------------------------------------------------------

    if args.list_models:

        print(
            "Available TypeSafe models:"
        )

        for item in models:
            print(
                f"  {item['name']}"
                f" | {item['release_date']}"
                f" | {item['description']}"
            )

        return


    model = choose_model(
        models
    )


    # --------------------------------------------------------
    # Load frozen experiment data
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    experiment_name = (
        "jev_zero_shot_v2"
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
    # Resume / restart
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
                "confidence",
                "probabilities",
                "latency_ms",
                "response_model",
                "input_tokens",
                "output_tokens",
                "raw_response",
            ]
        )

        completed_ids = set()


    # --------------------------------------------------------
    # Report configuration
    # --------------------------------------------------------

    print("=" * 70)
    print("JEV ZERO-SHOT — V2")
    print("=" * 70)

    print(
        f"Requested model : {model}"
    )

    print(
        f"Test examples   : "
        f"{len(test_df):,}"
    )

    print(
        f"V2 classes      : "
        f"{len(classes)}"
    )

    print(
        "New V2 labels   : 0"
    )

    print()


    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    total = len(test_df)

    for _, row in test_df.iterrows():

        example_id = row["id"]

        if example_id in completed_ids:
            continue


        result = classify(
            headers=headers,
            model=model,
            customer_text=str(
                row["text"]
            ),
            classes=classes,
        )


        prediction = (
            result["prediction"]
        )


        # Jev should only return a choice
        # defined in criteria.
        if prediction not in valid_labels:
            raise ValueError(
                "Jev returned an unexpected "
                f"choice: {prediction}"
            )


        new_row = pd.DataFrame(
            [
                {
                    "id":
                        example_id,

                    "prediction":
                        prediction,

                    "confidence":
                        result[
                            "confidence"
                        ],

                    "probabilities":
                        json.dumps(
                            result[
                                "probabilities"
                            ],
                            sort_keys=True,
                        ),

                    "latency_ms":
                        result[
                            "latency_ms"
                        ],

                    "response_model":
                        result[
                            "response_model"
                        ],

                    "input_tokens":
                        result[
                            "input_tokens"
                        ],

                    "output_tokens":
                        result[
                            "output_tokens"
                        ],

                    "raw_response":
                        json.dumps(
                            result[
                                "raw_response"
                            ],
                            sort_keys=True,
                        ),
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


        # Persist every result so an interrupted
        # API run can resume without paying again.
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
    # Keep only this run's requested IDs
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
        .copy()
    )


    # --------------------------------------------------------
    # Prediction file for evaluator
    # --------------------------------------------------------

    final_df[
        [
            "id",
            "prediction",
        ]
    ].to_csv(
        predictions_file,
        index=False,
    )


    # --------------------------------------------------------
    # Timing + token + cost statistics
    # --------------------------------------------------------

    latencies = (
        final_df["latency_ms"]
        .astype(float)
        .to_numpy()
    )


    total_input_tokens = int(
        final_df["input_tokens"]
        .astype(int)
        .sum()
    )

    total_output_tokens = int(
        final_df["output_tokens"]
        .astype(int)
        .sum()
    )


    estimated_input_cost_usd = (
        total_input_tokens
        / 1_000_000
        * INPUT_PRICE_PER_MILLION
    )


    actual_models = sorted(
        final_df["response_model"]
        .dropna()
        .unique()
        .tolist()
    )


    metadata = {
        "method":
            "jev_zero_shot",

        "task":
            "v2",

        "requested_model":
            model,

        "response_models":
            actual_models,

        "new_v2_labels":
            0,

        "test_examples":
            int(
                len(final_df)
            ),

        "latency_ms": {
            "mean":
                float(
                    np.mean(
                        latencies
                    )
                ),

            "p50":
                float(
                    np.percentile(
                        latencies,
                        50,
                    )
                ),

            "p95":
                float(
                    np.percentile(
                        latencies,
                        95,
                    )
                ),
        },

        "usage": {
            "input_tokens":
                total_input_tokens,

            "output_tokens":
                total_output_tokens,
        },

        "pricing": {
            "input_usd_per_million":
                INPUT_PRICE_PER_MILLION,

            "estimated_input_cost_usd":
                float(
                    estimated_input_cost_usd
                ),
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
        f"Predictions       : "
        f"{len(final_df):,}"
    )

    print(
        f"Response model(s) : "
        f"{actual_models}"
    )

    print(
        f"Mean latency      : "
        f"{metadata['latency_ms']['mean']:.2f} ms"
    )

    print(
        f"p50 latency       : "
        f"{metadata['latency_ms']['p50']:.2f} ms"
    )

    print(
        f"p95 latency       : "
        f"{metadata['latency_ms']['p95']:.2f} ms"
    )

    print(
        f"Input tokens      : "
        f"{total_input_tokens:,}"
    )

    print(
        f"Estimated cost    : "
        f"${estimated_input_cost_usd:.6f}"
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
