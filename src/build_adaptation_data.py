from pathlib import Path

import pandas as pd
import yaml


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TAXONOMY_FILE = ROOT / "config" / "taxonomy.yaml"
PREPARED_DIR = ROOT / "data" / "prepared"

TRAIN_GT_FILE = PREPARED_DIR / "train_ground_truth.csv"

ADAPTATION_DIR = PREPARED_DIR / "adaptation"

RANDOM_SEED = 42
NUM_DRAWS = 5

SMALL_BUDGET = 20
LARGE_BUDGET = 100


# ============================================================
# Load data and taxonomy
# ============================================================

with TAXONOMY_FILE.open("r", encoding="utf-8") as f:
    taxonomy = yaml.safe_load(f)

mapping = taxonomy["mapping"]

train_df = pd.read_csv(TRAIN_GT_FILE)


# ============================================================
# Determine V1 -> V2 transition automatically
#
# Do not hard-code clean/impure queues.
# Derive them from taxonomy.yaml.
# ============================================================

transition = {}

for intent, item in mapping.items():
    v1 = item["v1"]
    v2 = item["v2"]

    transition.setdefault(v1, set())
    transition[v1].add(v2)


clean_v1 = {
    v1
    for v1, destinations in transition.items()
    if len(destinations) == 1
}

impure_v1 = {
    v1
    for v1, destinations in transition.items()
    if len(destinations) > 1
}


# ============================================================
# Determine affected V2 queues
#
# These are V2 queues receiving examples from impure V1 queues.
# ============================================================

affected_v2 = set()

for v1 in impure_v1:
    affected_v2.update(transition[v1])


print("Clean V1 queues:")
for queue in sorted(clean_v1):
    print(f"  {queue}")

print("\nImpure V1 queues:")
for queue in sorted(impure_v1):
    print(f"  {queue}")

print("\nAffected V2 queues:")
for queue in sorted(affected_v2):
    print(f"  {queue}")


# ============================================================
# 1. Reusable historical data
#
# These examples were originally labelled using V1.
#
# Because a clean V1 queue maps entirely to one V2 queue,
# the V2 label can be inferred without acquiring a new label.
# ============================================================

clean_history = train_df[
    train_df["v1_label"].isin(clean_v1)
].copy()


# Keep the V2 label because it is deterministically implied by V1.
clean_history = clean_history[
    [
        "id",
        "text",
        "v1_label",
        "v2_label",
    ]
]


# Safety: each clean V1 class must map to exactly one V2 class.
for v1 in clean_v1:
    subset = clean_history[
        clean_history["v1_label"] == v1
    ]

    assert subset["v2_label"].nunique() == 1


clean_history_file = (
    PREPARED_DIR / "clean_v1_reusable.csv"
)

clean_history.to_csv(
    clean_history_file,
    index=False,
)


# ============================================================
# 2. Candidate pool for NEW V2 labels
#
# Historical examples from impure V1 queues cannot be
# deterministically relabelled.
#
# We use the underlying dataset ground truth only to simulate
# which examples the enterprise would newly label under V2.
#
# original_intent must NOT become a model feature.
# ============================================================

candidate_pool = train_df[
    train_df["v1_label"].isin(impure_v1)
].copy()


# All candidate examples should map to affected V2 queues.
assert set(candidate_pool["v2_label"].unique()).issubset(
    affected_v2
)


# ============================================================
# Verify enough examples exist for every V2 queue
# ============================================================

print("\nAvailable adaptation candidates:")
print("-" * 60)

for queue in sorted(affected_v2):
    count = (
        candidate_pool["v2_label"] == queue
    ).sum()

    print(f"{queue:<25} {count:>5}")

    if count < LARGE_BUDGET:
        raise ValueError(
            f"{queue} has only {count} candidate examples; "
            f"need at least {LARGE_BUDGET}."
        )


# ============================================================
# 3. Create five independent draws
#
# For every draw:
#
#   sample 100 examples / affected V2 queue
#
#   then:
#
#       20-example set = first 20 from the sampled 100
#
# Therefore:
#
#       v2_20 ⊂ v2_100
# ============================================================

ADAPTATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


for draw_index in range(NUM_DRAWS):

    draw_number = draw_index + 1

    draw_dir = (
        ADAPTATION_DIR
        / f"draw_{draw_number:02d}"
    )

    draw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    large_samples = []
    small_samples = []

    for queue_index, queue in enumerate(
        sorted(affected_v2)
    ):

        queue_pool = candidate_pool[
            candidate_pool["v2_label"] == queue
        ]

        # Different deterministic seed per draw and queue.
        seed = (
            RANDOM_SEED
            + draw_index * 100
            + queue_index
        )

        sampled_100 = queue_pool.sample(
            n=LARGE_BUDGET,
            random_state=seed,
        )

        # 20 is explicitly nested inside 100.
        sampled_20 = sampled_100.iloc[
            :SMALL_BUDGET
        ].copy()

        large_samples.append(sampled_100)
        small_samples.append(sampled_20)

    adaptation_100 = pd.concat(
        large_samples,
        ignore_index=True,
    )

    adaptation_20 = pd.concat(
        small_samples,
        ignore_index=True,
    )


    # ========================================================
    # Remove hidden BANKING77 intent information.
    #
    # Model-training files should contain only:
    #
    # id
    # text
    # v2_label
    #
    # v1_label is also unnecessary for training.
    # ========================================================

    adaptation_100 = adaptation_100[
        [
            "id",
            "text",
            "v2_label",
        ]
    ]

    adaptation_20 = adaptation_20[
        [
            "id",
            "text",
            "v2_label",
        ]
    ]


    # ========================================================
    # Safety checks
    # ========================================================

    expected_20 = (
        len(affected_v2)
        * SMALL_BUDGET
    )

    expected_100 = (
        len(affected_v2)
        * LARGE_BUDGET
    )

    assert len(adaptation_20) == expected_20
    assert len(adaptation_100) == expected_100


    # Verify nested sampling.
    assert set(adaptation_20["id"]).issubset(
        set(adaptation_100["id"])
    )


    # Verify exact budget per V2 class.
    counts_20 = (
        adaptation_20["v2_label"]
        .value_counts()
    )

    counts_100 = (
        adaptation_100["v2_label"]
        .value_counts()
    )

    for queue in affected_v2:
        assert counts_20[queue] == SMALL_BUDGET
        assert counts_100[queue] == LARGE_BUDGET


    # ========================================================
    # Save
    # ========================================================

    adaptation_20.to_csv(
        draw_dir / "v2_20.csv",
        index=False,
    )

    adaptation_100.to_csv(
        draw_dir / "v2_100.csv",
        index=False,
    )


# ============================================================
# Report
# ============================================================

print()
print("=" * 70)
print("V2 ADAPTATION DATA CREATED")
print("=" * 70)

print(
    f"Reusable clean V1 examples : "
    f"{len(clean_history):,}"
)

print(
    f"Affected V2 queues         : "
    f"{len(affected_v2)}"
)

print(
    f"20-label set size          : "
    f"{len(affected_v2) * SMALL_BUDGET:,}"
)

print(
    f"100-label set size         : "
    f"{len(affected_v2) * LARGE_BUDGET:,}"
)

print(
    f"Random draws               : "
    f"{NUM_DRAWS}"
)


print()
print("Reusable historical data:")
print("-" * 70)

print(
    clean_history["v2_label"]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("Example draw_01 / 20-label distribution:")
print("-" * 70)

example = pd.read_csv(
    ADAPTATION_DIR
    / "draw_01"
    / "v2_20.csv"
)

print(
    example["v2_label"]
    .value_counts()
    .sort_index()
    .to_string()
)


print()
print("Files created:")
print("-" * 70)

print("  data/prepared/clean_v1_reusable.csv")

for draw_index in range(NUM_DRAWS):
    draw_number = draw_index + 1

    print(
        f"  data/prepared/adaptation/"
        f"draw_{draw_number:02d}/v2_20.csv"
    )

    print(
        f"  data/prepared/adaptation/"
        f"draw_{draw_number:02d}/v2_100.csv"
    )