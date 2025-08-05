"""
Modernized utilities for AutoBool training with HuggingFace datasets support.

This module provides utilities that work with both local JSON files and 
HuggingFace datasets, replacing the old JSON-only approach.
"""

import os
import json
import copy
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import random

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Add parent directory for logging
import sys
sys.path.append('..')
from utils.logging_config import get_logger

logger = get_logger("autobool.utils")


def form_completion(boolean_query: str, sample_out_format: List[Dict]) -> List[Dict]:
    """
    Format a completion with the given boolean query.
    
    Args:
        boolean_query: The boolean query string
        sample_out_format: Template for completion format
    
    Returns:
        Formatted completion
    """
    completion = copy.deepcopy(sample_out_format)
    completion[0]["content"] = completion[0]["content"].format(boolean=boolean_query.strip())
    return completion


def create_split_dataset(data: List[Dict], seed: int = 42, test_size: float = 0.5) -> DatasetDict:
    """
    Create temporal train/test/temporal1000 splits from data.
    
    Args:
        data: List of data samples
        seed: Random seed for reproducibility
        test_size: Fraction of data for test set
    
    Returns:
        DatasetDict with train/test/temporal1000 splits
    """
    logger.info(f"Creating dataset splits with test_size={test_size}, seed={seed}")
    random.seed(seed)

    # Parse and sort by 'max_date' (format: 'YYYY/MM/DD')
    for item in data:
        date_str = item.get("max-date") or item.get("last_ref_date")
        if date_str:
            try:
                item["_parsed_date"] = datetime.strptime(date_str, "%Y/%m/%d")
            except ValueError:
                logger.warning(f"Invalid date format: {date_str}, using default")
                item["_parsed_date"] = datetime.strptime("2025/06/24", "%Y/%m/%d")
        else:
            item["_parsed_date"] = datetime.strptime("2025/06/24", "%Y/%m/%d")

    data_sorted = sorted(data, key=lambda x: x["_parsed_date"])

    # Compute split index
    split_index = int(len(data_sorted) * (1 - test_size))
    train_data = data_sorted[:split_index]
    test_data = data_sorted[split_index:]

    logger.info(f"Train: {train_data[0]['max-date']} → {train_data[-1]['max-date']} ({len(train_data)} samples)")
    logger.info(f"Test:  {test_data[0]['max-date']} → {test_data[-1]['max-date']} ({len(test_data)} samples)")

    # Create deterministic temporal1000 split
    cutoff_date = datetime.strptime("2024/11/01", "%Y/%m/%d")
    temporal_candidates = [item for item in data_sorted if item["_parsed_date"] > cutoff_date]
    
    # Remove non-serializable keys (like _parsed_date) before sorting
    for item in temporal_candidates:
        item.pop("_parsed_date", None)
    
    # Ensure consistent order before sampling
    temporal_candidates = sorted(temporal_candidates, key=lambda x: x.get("topicid", ""))

    if len(temporal_candidates) >= 1000:
        temporal_1000 = random.sample(temporal_candidates, 1000)
        logger.info(f"Created temporal1000 split with 1000 samples")
    else:
        logger.warning(f"Only {len(temporal_candidates)} samples found after 2024/11/01. Using all of them.")
        temporal_1000 = temporal_candidates

    # Clean up the helper field
    for item in data_sorted:
        item.pop("_parsed_date", None)

    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data),
        "temporal1000": Dataset.from_list(temporal_1000),
    })


def build_prompt(
    title: str, 
    sample_format: Optional[List[Dict]] = None, 
    tokenizer: Optional[AutoTokenizer] = None, 
    enable_thinking: bool = False
) -> Union[List[Dict], str]:
    """
    Build a prompt from a title and format template.
    
    Args:
        title: The topic title
        sample_format: Template for prompt format
        tokenizer: Optional tokenizer for chat template
        enable_thinking: Whether to enable thinking tags
    
    Returns:
        Formatted prompt (list of dicts or string if tokenized)
    """
    if sample_format is None:
        raise ValueError("sample_format is required")
    
    sample = copy.deepcopy(sample_format)
    sample[1]["content"] = sample[1]["content"].format(topic=title.strip())
    
    if tokenizer:
        sample = tokenizer.apply_chat_template(
            sample, 
            add_generation_prompt=True, 
            enable_thinking=enable_thinking, 
            tokenize=False
        )
    
    return sample


def load_jsonl_dataset(
    dataset_folder: str, 
    completion: bool = False, 
    sample_format: Optional[List[Dict]] = None, 
    sample_out_format: Optional[List[Dict]] = None, 
    enable_thinking: bool = True, 
    tokenizer_path: Optional[str] = None
) -> List[Dict]:
    """
    Load dataset from JSONL files in a folder.
    
    Args:
        dataset_folder: Path to folder containing JSONL files
        completion: Whether to include completion format
        sample_format: Template for input format
        sample_out_format: Template for output format
        enable_thinking: Enable thinking tags
        tokenizer_path: Path to tokenizer
    
    Returns:
        List of processed data samples
    """
    logger.info(f"Loading JSONL dataset from: {dataset_folder}")
    
    data = []
    files = [f for f in os.listdir(dataset_folder) if f.endswith('.json') or f.endswith('.jsonl')]
    
    if not enable_thinking:
        if not tokenizer_path:
            raise ValueError("tokenizer_path must be provided when enable_thinking is False.")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = None

    for file_name in tqdm(files, desc="Processing files"):
        file_path = os.path.join(dataset_folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Reading {file_name}", leave=False):
                try:
                    record = json.loads(line.strip())
                    
                    # Extract topic ID (either topicid or pmid)
                    if "topicid" in record:
                        topicid = record["topicid"]
                    elif "pmid" in record:
                        topicid = record["pmid"]
                    else:
                        topicid = ""

                    # Only process records with title and references
                    if record.get("title") and (record.get("references-pmids") or record.get("results-section-pmids")):
                        
                        prompt = build_prompt(
                            record["title"], 
                            sample_format=sample_format, 
                            tokenizer=tokenizer, 
                            enable_thinking=enable_thinking
                        )

                        # Extract date information
                        min_date = record.get("min-date", "")
                        max_date = record.get("max-date", "")
                        first_ref_date = record.get("first_ref_date", "")
                        last_ref_date = record.get("last_ref_date", "")

                        # Extract ground truth references
                        ground_truth = record.get("results-section-pmids", record.get("references-pmids"))

                        current_dict = {
                            "prompt": prompt,
                            "topicid": topicid,
                            "min-date": min_date,
                            "max-date": max_date,
                            "first_ref_date": first_ref_date,
                            "last_ref_date": last_ref_date,
                            "ground_truth": ground_truth
                        }
                        data.append(current_dict)
                        
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"Loaded {len(data)} samples from {len(files)} files")
    return data


def process_and_push(
    folder_path: str, 
    hf_name: str, 
    seed: int = 42, 
    completion: bool = False, 
    sample_format: Optional[List[Dict]] = None, 
    sample_out_format: Optional[List[Dict]] = None, 
    enable_thinking: bool = True, 
    tokenizer_path: Optional[str] = None, 
    split: bool = True
) -> None:
    """
    Process JSONL data and push to HuggingFace Hub.
    
    Args:
        folder_path: Path to folder containing JSONL files
        hf_name: HuggingFace dataset name
        seed: Random seed for splits
        completion: Include completion format
        sample_format: Input format template
        sample_out_format: Output format template
        enable_thinking: Enable thinking tags
        tokenizer_path: Path to tokenizer
        split: Whether to create train/test splits
    """
    logger.info(f"Processing and pushing dataset: {hf_name}")
    
    data = load_jsonl_dataset(
        folder_path, 
        completion=completion, 
        sample_format=sample_format, 
        sample_out_format=sample_out_format, 
        enable_thinking=enable_thinking, 
        tokenizer_path=tokenizer_path
    )
    
    logger.info(f"Total samples: {len(data)}")
    
    if split:
        split_dataset = create_split_dataset(data, seed=seed)
        logger.info(f"Dataset splits: {split_dataset}")
        split_dataset.push_to_hub(hf_name, private=True)
        logger.info(f"Successfully pushed split dataset to: {hf_name}")
    else:
        dataset = Dataset.from_list(data)
        logger.info(f"Single dataset: {dataset}")
        dataset.push_to_hub(hf_name, private=True)
        logger.info(f"Successfully pushed dataset to: {hf_name}")


# Modern HuggingFace dataset utilities
def load_hf_dataset(
    dataset_name: str, 
    split: Optional[str] = None,
    streaming: bool = False,
    trust_remote_code: bool = False
) -> Union[Dataset, DatasetDict]:
    """
    Load dataset from HuggingFace Hub.
    
    Args:
        dataset_name: Name of HuggingFace dataset
        split: Specific split to load (train, test, etc.)
        streaming: Whether to use streaming mode
        trust_remote_code: Whether to trust remote code
    
    Returns:
        Dataset or DatasetDict
    """
    logger.info(f"Loading HuggingFace dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(
            dataset_name, 
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code
        )
        
        if split:
            logger.info(f"Loaded {split} split with {len(dataset)} samples")
        else:
            logger.info(f"Loaded dataset with splits: {list(dataset.keys())}")
        
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise


def get_dataset_info(dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
    """
    Get information about a dataset.
    
    Args:
        dataset: HuggingFace Dataset or DatasetDict
    
    Returns:
        Dictionary with dataset information
    """
    info = {}
    
    if isinstance(dataset, DatasetDict):
        info["type"] = "DatasetDict"
        info["splits"] = {}
        for split_name, split_data in dataset.items():
            info["splits"][split_name] = {
                "num_samples": len(split_data),
                "features": list(split_data.features.keys())
            }
    else:
        info["type"] = "Dataset"
        info["num_samples"] = len(dataset)
        info["features"] = list(dataset.features.keys())
    
    return info


def sample_dataset(dataset: Dataset, n_samples: int = 5, seed: int = 42) -> List[Dict]:
    """
    Sample a few examples from a dataset.
    
    Args:
        dataset: HuggingFace Dataset
        n_samples: Number of samples to return
        seed: Random seed
    
    Returns:
        List of sample records
    """
    if len(dataset) < n_samples:
        n_samples = len(dataset)
    
    # Get random indices
    random.seed(seed)
    indices = random.sample(range(len(dataset)), n_samples)
    
    samples = []
    for idx in indices:
        samples.append(dataset[idx])
    
    return samples


# Legacy compatibility function
def load_dataset_legacy(file_path: str) -> List[Dict]:
    """
    Legacy function to load from single JSON file.
    Maintained for backward compatibility.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        List of data samples
    """
    logger.warning("Using legacy JSON loading. Consider migrating to HuggingFace datasets.")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} samples from legacy JSON file")
    return data