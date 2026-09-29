"""Download and inspect the Parquet snapshot of BANKING77 from the Hub."""

from pathlib import Path

from datasets import load_dataset


DATASET_ID = "PolyAI/banking77"
DATASET_REVISION = "062c492e118a36c10e8c8bf4308fce0f63152d3e"


def prepare(raw_dir: Path = Path("data/raw")):
    """Download the script-free, Parquet-backed Hub snapshot to ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        cache_dir=str(raw_dir / ".hf_cache"),
    )
    dataset.save_to_disk(str(raw_dir / "banking77"))
    return dataset


def main() -> None:
    dataset = prepare()
    labels = dataset["train"].features["label"].names
    print(dataset)
    print("\nFirst training example:")
    print(dataset["train"][0])
    print(f"\nNumber of labels: {len(labels)}")
    for index, label in enumerate(labels):
        print(f"{index:2d}: {label}")


if __name__ == "__main__":
    main()
