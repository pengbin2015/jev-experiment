"""Validate the frozen BANKING77 V1/V2 mapping."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

from datasets import DatasetDict, load_from_disk
import yaml


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_FILE = ROOT / "config" / "taxonomy.yaml"
DATASET_DIR = ROOT / "data" / "raw" / "banking77"


def validate(path: Path, dataset: DatasetDict | None = None) -> list[str]:
    """Return schema and label coverage errors for a taxonomy file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return ["taxonomy must be a mapping"]

    errors: list[str] = []
    if data.get("version") != "1.0":
        errors.append('version must be "1.0"')
    if data.get("dataset") != "PolyAI/banking77":
        errors.append('dataset must be "PolyAI/banking77"')

    queues: dict[str, dict] = {}
    for stage, expected_count in (("v1", 7), ("v2", 9)):
        stage_queues = data.get(stage)
        if not isinstance(stage_queues, dict):
            errors.append(f"{stage} must be a queue mapping")
            continue
        queues[stage] = stage_queues
        if len(stage_queues) != expected_count:
            errors.append(f"{stage} must define {expected_count} queues")
        for name, details in stage_queues.items():
            if not isinstance(details, dict) or not isinstance(details.get("description"), str) or not details["description"].strip():
                errors.append(f"{stage}.{name} needs a description")

    mapping = data.get("mapping")
    if not isinstance(mapping, dict):
        return errors + ["mapping must be an intent mapping"]
    if len(mapping) != 77:
        errors.append(f"mapping must contain 77 intents; found {len(mapping)}")

    for intent, assignment in mapping.items():
        if not isinstance(assignment, dict):
            errors.append(f"{intent} must map to V1 and V2 queues")
            continue
        for stage in ("v1", "v2"):
            if assignment.get(stage) not in queues.get(stage, {}):
                errors.append(f"{intent} references an unknown {stage} queue")
        change = assignment.get("change")
        if change not in {"split", "move", "unchanged"}:
            errors.append(f"{intent} has an invalid change type")
        elif (assignment.get("v1") == assignment.get("v2")) != (change == "unchanged"):
            errors.append(f"{intent} has an inconsistent change type")

    # A fresh checkout can validate structure without downloading the dataset.
    if dataset is None and DATASET_DIR.exists():
        dataset = load_from_disk(str(DATASET_DIR))
    if dataset is not None:
        labels = set(dataset["train"].features["label"].names)
        missing = labels - set(mapping)
        extra = set(mapping) - labels
        if missing:
            errors.append(f"missing source labels: {', '.join(sorted(missing))}")
        if extra:
            errors.append(f"unknown source labels: {', '.join(sorted(extra))}")

    return errors


def print_distribution(title: str, queues: dict, train: Counter, test: Counter) -> None:
    """Print the split sizes for one taxonomy version."""
    print(f"\n{title}")
    print(f"{'Queue':<25} {'Train':>8} {'Test':>8} {'Total':>8} {'Train %':>9}")
    total_train = sum(train.values())
    total_test = sum(test.values())
    for queue in queues:
        train_count = train[queue]
        test_count = test[queue]
        print(
            f"{queue:<25} {train_count:>8} {test_count:>8} "
            f"{train_count + test_count:>8} {train_count / total_train:>8.1%}"
        )
    print(f"{'TOTAL':<25} {total_train:>8} {total_test:>8} {total_train + total_test:>8}")


def print_report(taxonomy: dict, dataset: DatasetDict) -> None:
    """Report queue distributions and the V1 to V2 transition."""
    mapping = taxonomy["mapping"]
    labels = dataset["train"].features["label"].names

    for version in ("v1", "v2"):
        counts = {
            split: Counter(
                mapping[labels[example["label"]]][version]
                for example in dataset[split]
            )
            for split in ("train", "test")
        }
        print_distribution(
            f"{version.upper()} DISTRIBUTION",
            taxonomy[version],
            counts["train"],
            counts["test"],
        )

    transition: dict[str, set[str]] = defaultdict(set)
    for assignment in mapping.values():
        transition[assignment["v1"]].add(assignment["v2"])

    print("\nV1 -> V2 TRANSITION")
    impure: list[str] = []
    for queue in taxonomy["v1"]:
        destinations = sorted(transition[queue])
        status = "CLEAN" if len(destinations) == 1 else "IMPURE"
        if status == "IMPURE":
            impure.append(queue)
        print(f"{queue:<20} -> {', '.join(destinations)} [{status}]")

    affected = sorted({destination for queue in impure for destination in transition[queue]})
    print(f"\nAffected V2 queues: {', '.join(affected)}")


if __name__ == "__main__":
    dataset = load_from_disk(str(DATASET_DIR)) if DATASET_DIR.exists() else None
    errors = validate(TAXONOMY_FILE, dataset)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        sys.exit(1)
    if dataset is None:
        print("Taxonomy structure is valid. Run prepare_data.py to check source labels and distributions.")
    else:
        print("Taxonomy is valid: 77 intents mapped to 7 V1 and 9 V2 queues.")
        print_report(yaml.safe_load(TAXONOMY_FILE.read_text(encoding="utf-8")), dataset)
