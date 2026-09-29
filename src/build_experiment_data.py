from pathlib import Path

import pandas as pd
import yaml
from datasets import load_from_disk


RANDOM_SEED = 42
TEST_SIZE = 600

ROOT = Path(__file__).resolve().parents[1]

TAXONOMY_FILE = ROOT / "config" / "taxonomy.yaml"
RAW_DATASET_DIR = ROOT / "data" / "raw" / "banking77"
OUTPUT_DIR = ROOT / "data" / "prepared"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load frozen taxonomy
# ------------------------------------------------------------

with TAXONOMY_FILE.open("r", encoding="utf-8") as f:
    taxonomy = yaml.safe_load(f)

mapping = taxonomy["mapping"]


# ------------------------------------------------------------
# Load frozen local BANKING77 snapshot
# ------------------------------------------------------------

dataset = load_from_disk(str(RAW_DATASET_DIR))

intent_names = dataset["train"].features["label"].names


# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------

assert len(intent_names) == 77
assert set(intent_names) == set(mapping.keys())

assert len(dataset["train"]) == 10003
assert len(dataset["test"]) == 3080


# ------------------------------------------------------------
# Convert HF Dataset split -> experiment ground truth
# ------------------------------------------------------------

def build_ground_truth(split_name: str) -> pd.DataFrame:
    split = dataset[split_name]

    rows = []

    for i, example in enumerate(split):
        original_intent = intent_names[example["label"]]

        rows.append(
            {
                "id": f"{split_name}_{i:05d}",
                "text": example["text"],
                "original_intent": original_intent,
                "v1_label": mapping[original_intent]["v1"],
                "v2_label": mapping[original_intent]["v2"],
            }
        )

    return pd.DataFrame(rows)


train_gt = build_ground_truth("train")
test_gt = build_ground_truth("test")


# ------------------------------------------------------------
# Validate prepared ground truth
# ------------------------------------------------------------

assert len(train_gt) == 10003
assert len(test_gt) == 3080

assert train_gt["original_intent"].nunique() == 77
assert test_gt["original_intent"].nunique() == 77

assert train_gt["v1_label"].nunique() == len(taxonomy["v1"])
assert test_gt["v1_label"].nunique() == len(taxonomy["v1"])

assert train_gt["v2_label"].nunique() == len(taxonomy["v2"])
assert test_gt["v2_label"].nunique() == len(taxonomy["v2"])

assert not train_gt.isnull().any().any()
assert not test_gt.isnull().any().any()


# ------------------------------------------------------------
# Save complete evaluator-side ground truth
# ------------------------------------------------------------

train_gt.to_csv(
    OUTPUT_DIR / "train_ground_truth.csv",
    index=False,
)

test_gt.to_csv(
    OUTPUT_DIR / "test_ground_truth.csv",
    index=False,
)


# ------------------------------------------------------------
# Frozen 600-example test set
#
# Approximate V2 distribution, with a minimum of 40 examples
# for smaller/new V2 queues.
# ------------------------------------------------------------

test_allocation = {
    "CARD_MANAGEMENT": 135,
    "DIGITAL_PAYMENTS": 78,
    "CASH_WITHDRAWAL": 40,
    "TRANSFERS": 83,
    "TOP_UPS": 91,
    "ACCOUNT_SERVICES": 40,
    "IDENTITY_COMPLIANCE": 40,
    "SECURITY_DISPUTES": 47,
    "CURRENCY": 46,
}

assert sum(test_allocation.values()) == TEST_SIZE


sampled_groups = []

for offset, (v2_class, n) in enumerate(test_allocation.items()):
    group = test_gt[test_gt["v2_label"] == v2_class]

    if len(group) < n:
        raise ValueError(
            f"{v2_class}: need {n} test examples, "
            f"but only {len(group)} are available."
        )

    sampled = group.sample(
        n=n,
        random_state=RANDOM_SEED + offset,
    )

    sampled_groups.append(sampled)


test_600_gt = pd.concat(
    sampled_groups,
    ignore_index=True,
)

test_600_gt = (
    test_600_gt
    .sample(frac=1, random_state=RANDOM_SEED)
    .reset_index(drop=True)
)

assert len(test_600_gt) == 600
assert test_600_gt["id"].is_unique


# ------------------------------------------------------------
# Evaluator-side frozen test set
# ------------------------------------------------------------

test_600_gt.to_csv(
    OUTPUT_DIR / "test_600_ground_truth.csv",
    index=False,
)


# ------------------------------------------------------------
# Model-facing test set
#
# Models receive ONLY id + text.
# ------------------------------------------------------------

test_600_inputs = test_600_gt[
    ["id", "text"]
].copy()

test_600_inputs.to_csv(
    OUTPUT_DIR / "test_600_inputs.csv",
    index=False,
)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print()
print("=" * 70)
print("EXPERIMENT DATA CREATED")
print("=" * 70)

print(f"Training ground truth : {len(train_gt):,}")
print(f"Full test ground truth: {len(test_gt):,}")
print(f"Frozen test set       : {len(test_600_gt):,}")

print()
print("Frozen 600-test V2 distribution")
print("-" * 70)

print(
    test_600_gt["v2_label"]
    .value_counts()
    .reindex(test_allocation.keys())
    .to_string()
)

print()
print("Frozen 600-test V1 distribution")
print("-" * 70)

print(
    test_600_gt["v1_label"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print("Files created")
print("-" * 70)

for filename in [
    "train_ground_truth.csv",
    "test_ground_truth.csv",
    "test_600_ground_truth.csv",
    "test_600_inputs.csv",
]:
    print(f"  data/prepared/{filename}")

print()
print(
    "Model code should read test_600_inputs.csv, "
    "not test_600_ground_truth.csv."
)