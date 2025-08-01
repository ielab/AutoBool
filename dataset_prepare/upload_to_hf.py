#!/usr/bin/env python3
"""
Upload JSONL file to HuggingFace Dataset Hub with temporal splitting.

This script reads a JSONL file containing medical literature data and uploads it
to HuggingFace Dataset Hub with train/test/temporal1000 splits based on publication dates.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                print(f"Line content: {line}")
                continue

    print(f"Loaded {len(data)} items from {file_path}")
    return data


def validate_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and clean the data."""
    valid_data = []

    for i, item in enumerate(data):
        # Check required fields
        required_fields = ["pmid", "title"]
        missing_fields = [field for field in required_fields if field not in item or not item[field]]

        if missing_fields:
            print(f"Skipping item {i}: missing required fields {missing_fields}")
            continue

        # Ensure lists are properly formatted
        list_fields = ["references-pmids", "results-section-pmids"]
        for field in list_fields:
            if field in item and item[field] is not None:
                if not isinstance(item[field], list):
                    print(f"Warning: {field} in item {i} is not a list, converting")
                    item[field] = [item[field]] if item[field] else []
            else:
                item[field] = []

        # Add topicid if not present (using pmid as fallback)
        if "topicid" not in item:
            item["topicid"] = item["pmid"]

        valid_data.append(item)

    print(f"Validated {len(valid_data)} items ({len(data) - len(valid_data)} items removed)")
    return valid_data


def create_split_dataset(data: List[dict], seed: int = 42, test_size: float = 0.5) -> DatasetDict:
    """
    Create train/test/temporal1000 splits based on publication dates.

    Args:
        data: List of data items
        seed: Random seed for reproducible splits
        test_size: Proportion of data to use for test set

    Returns:
        DatasetDict with train, test, and temporal1000 splits
    """
    random.seed(seed)

    # Parse and sort by 'max_date' (format: 'YYYY/MM/DD')
    for item in data:
        date_str = item.get("max-date") or item.get("last_ref_date")
        if date_str:
            try:
                item["_parsed_date"] = datetime.strptime(date_str, "%Y/%m/%d")
            except ValueError:
                print(f"Warning: Invalid date format '{date_str}' for item {item.get('pmid', 'unknown')}")
                item["_parsed_date"] = datetime.strptime("2025/06/24", "%Y/%m/%d")
        else:
            item["_parsed_date"] = datetime.strptime("2025/06/24", "%Y/%m/%d")

    data_sorted = sorted(data, key=lambda x: x["_parsed_date"])

    # Compute split index
    split_index = int(len(data_sorted) * (1 - test_size))
    train_data = data_sorted[:split_index]
    test_data = data_sorted[split_index:]

    print(
        f"Train: {train_data[0].get('max-date', 'N/A')} → {train_data[-1].get('max-date', 'N/A')} ({len(train_data)} samples)")
    print(
        f"Test:  {test_data[0].get('max-date', 'N/A')} → {test_data[-1].get('max-date', 'N/A')} ({len(test_data)} samples)")

    # Create deterministic temporal1000 split
    cutoff_date = datetime.strptime("2024/11/01", "%Y/%m/%d")
    temporal_candidates = [item for item in data_sorted if item["_parsed_date"] > cutoff_date]

    # Remove non-serializable keys (like _parsed_date) before sorting
    for item in temporal_candidates:
        item.pop("_parsed_date", None)

    # Ensure consistent order before sampling
    temporal_candidates = sorted(temporal_candidates, key=lambda x: x["topicid"])

    if len(temporal_candidates) >= 1000:
        temporal_1000 = random.sample(temporal_candidates, 1000)
        print(f"Temporal1000: Selected 1000 samples from {len(temporal_candidates)} candidates after 2024/11/01")
    else:
        print(f"[WARN] Only {len(temporal_candidates)} samples found after 2024/11/01. Using all of them.")
        temporal_1000 = temporal_candidates

    # Clean up the helper field from all data
    for item in data_sorted:
        item.pop("_parsed_date", None)

    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data),
        "temporal1000": Dataset.from_list(temporal_1000),
    })


def create_dataset_card(dataset_name: str, num_train: int, num_test: int, num_temporal: int) -> str:
    """Create a README.md content for the dataset."""
    return f"""# {dataset_name}

## Dataset Description

This dataset contains medical literature data for training Boolean query generation models. The data includes PubMed articles with their associated metadata, references, and result section PMIDs.

## Dataset Structure

### Data Fields

- `pmid`: PubMed ID of the article
- `pmc-id`: PMC ID (if available)
- `title`: Article title
- `max-date`: Maximum publication date
- `references-pmids`: List of PMIDs referenced in the article
- `first_ref_date`: Date of first reference
- `last_ref_date`: Date of last reference
- `results-section-pmids`: List of PMIDs found in the results section
- `topicid`: Topic identifier (defaults to pmid if not present)

### Data Splits

- **Train**: {num_train:,} samples - Earlier publications for training
- **Test**: {num_test:,} samples - Later publications for evaluation  
- **Temporal1000**: {num_temporal:,} samples - Recent publications (after Nov 1, 2024) for temporal evaluation

## Usage

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("{dataset_name}")

# Load specific split
train_data = load_dataset("{dataset_name}", split="train")
test_data = load_dataset("{dataset_name}", split="test")
temporal_data = load_dataset("{dataset_name}", split="temporal1000")
```

## Citation

If you use this dataset, please cite the original data sources and any relevant publications.
"""


def upload_to_huggingface(
        jsonl_file: str,
        dataset_name: str,
        private: bool = False,
        test_size: float = 0.5,
        seed: int = 42,
        push_to_hub: bool = True
):
    """
    Upload JSONL file to HuggingFace Dataset Hub.

    Args:
        jsonl_file: Path to the JSONL file
        dataset_name: HuggingFace dataset name (e.g., "username/dataset-name")
        private: Whether to create a private dataset
        test_size: Proportion of data for test split
        seed: Random seed for reproducible splits
        push_to_hub: Whether to actually push to hub (set False for testing)
    """
    print(f"Processing {jsonl_file} → {dataset_name}")
    print(f"Private: {private}, Test size: {test_size}, Seed: {seed}")
    print("-" * 60)

    # Load and validate data
    data = load_jsonl(jsonl_file)
    if not data:
        print("No valid data found. Exiting.")
        return

    data = validate_data(data)
    if not data:
        print("No valid data after validation. Exiting.")
        return

    # Create splits
    print("\nCreating dataset splits...")
    dataset_dict = create_split_dataset(data, seed=seed, test_size=test_size)

    # Print split information
    print(f"\nDataset splits created:")
    for split_name, split_data in dataset_dict.items():
        print(f"  {split_name}: {len(split_data):,} samples")

    # Preview first item from each split
    print(f"\nPreview of data structure:")
    for split_name, split_data in dataset_dict.items():
        if len(split_data) > 0:
            sample = split_data[0]
            print(f"\n{split_name.upper()} sample:")
            for key, value in sample.items():
                if isinstance(value, list):
                    print(f"  {key}: [{len(value)} items] {value[:3]}{'...' if len(value) > 3 else ''}")
                elif isinstance(value, str) and len(value) > 50:
                    print(f"  {key}: {value[:50]}...")
                else:
                    print(f"  {key}: {value}")

    if not push_to_hub:
        print("\nDry run complete. Set --push_to_hub to actually upload.")
        return

    try:
        # Push to hub
        print(f"\nUploading to HuggingFace Hub: {dataset_name}")
        dataset_dict.push_to_hub(
            dataset_name,
            private=private,
            commit_message="Initial dataset upload from JSONL"
        )

        # Create and upload README
        readme_content = create_dataset_card(
            dataset_name,
            len(dataset_dict["train"]),
            len(dataset_dict["test"]),
            len(dataset_dict["temporal1000"])
        )

        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=dataset_name,
            repo_type="dataset",
            commit_message="Add dataset card"
        )

        print(f"✅ Successfully uploaded dataset to: https://huggingface.co/datasets/{dataset_name}")

    except Exception as e:
        print(f"❌ Error uploading to HuggingFace Hub: {e}")
        print("Make sure you're logged in with `huggingface-cli login`")


def main():
    parser = argparse.ArgumentParser(
        description="Upload JSONL file to HuggingFace Dataset Hub with temporal splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "jsonl_file",
        type=str,
        help="Path to the JSONL file to upload"
    )

    parser.add_argument(
        "dataset_name",
        type=str,
        help="HuggingFace dataset name (e.g., 'username/dataset-name')"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private dataset"
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.5,
        help="Proportion of data to use for test split"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process data and create splits but don't upload to HuggingFace"
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.jsonl_file).exists():
        print(f"Error: File {args.jsonl_file} does not exist")
        return

    if not args.dataset_name or '/' not in args.dataset_name:
        print("Error: dataset_name should be in format 'username/dataset-name'")
        return

    if not (0 < args.test_size < 1):
        print("Error: test_size should be between 0 and 1")
        return

    # Upload dataset
    upload_to_huggingface(
        jsonl_file=args.jsonl_file,
        dataset_name=args.dataset_name,
        private=args.private,
        test_size=args.test_size,
        seed=args.seed,
        push_to_hub=not args.dry_run
    )


if __name__ == "__main__":
    main()