import os
import json
from datasets import Dataset, DatasetDict
from typing import List
from tqdm import tqdm
import copy
from reward import cached_retrieve_documents
from transformers import AutoTokenizer
import sys
def form_completion(boolean_query: str, sample_out_format: List[dict]) -> List[dict]:
    completion = copy.deepcopy(sample_out_format)
    completion[0]["content"] = completion[0]["content"].format(boolean=boolean_query.strip())
    return completion


from datetime import datetime
import random

def create_split_dataset(data: List[dict], seed: int = 42, test_size: float = 0.5) -> DatasetDict:
    random.seed(seed)

    # Parse and sort by 'max_date' (format: 'YYYY/MM/DD')
    for item in data:
        date_str = item.get("max-date") or item.get("last_ref_date")
        if date_str:
            item["_parsed_date"] = datetime.strptime(date_str, "%Y/%m/%d")
        else:
            item["_parsed_date"] = datetime.strptime("2025/06/24", "%Y/%m/%d")

    data_sorted = sorted(data, key=lambda x: x["_parsed_date"])

    # Compute split index
    split_index = int(len(data_sorted) * (1 - test_size))
    train_data = data_sorted[:split_index]
    test_data = data_sorted[split_index:]

    print(f"Train: {train_data[0]['max-date']} → {train_data[-1]['max-date']}")
    print(f"Test:  {test_data[0]['max-date']} → {test_data[-1]['max-date']}")

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
    else:
        print(f"[WARN] Only {len(temporal_candidates)} samples found after 2024/11/01. Using all of them.")
        temporal_1000 = temporal_candidates

    # Clean up the helper field
    for item in data_sorted:
        item.pop("_parsed_date", None)

    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data),
        "temporal1000": Dataset.from_list(temporal_1000),
    })




def process_and_push(folder_path: str, hf_name: str, seed: int = 42, completion: bool = False, sample_format: List[dict] = None, sample_out_format: List[dict] = None, enable_thinking: bool = True, tokenizer_path: str = None, split: bool=True) -> None:
    data = load_jsonl_dataset(folder_path, completion=completion, sample_format=sample_format, sample_out_format=sample_out_format, enable_thinking=enable_thinking, tokenizer_path=tokenizer_path)
    print(len(data))
    if split:
        split_dataset = create_split_dataset(data, seed=seed)
        print(split_dataset)
        split_dataset.push_to_hub(hf_name, private=True)
    else:
        dataset = Dataset.from_list(data)
        print(dataset)
        dataset.push_to_hub(hf_name, private=True)



def build_prompt(title: str, sample_format: List[dict] = None, tokenizer: AutoTokenizer = None, enable_thinking: bool = False) -> List[dict]:
    sample = copy.deepcopy(sample_format)
    sample[1]["content"] = sample[1]["content"].format(topic=title.strip())
    if tokenizer:
        sample = tokenizer.apply_chat_template(sample, add_generation_prompt=True, enable_thinking=enable_thinking, tokenize=False)
    return sample


def load_jsonl_dataset(dataset_folder: str, completion: bool = False, sample_format: List[dict] = None, sample_out_format: List[dict] = None, enable_thinking: bool=True, tokenizer_path: str=None) -> List[dict]:
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
                    # topicid is  either topicid or pmid
                    if "topicid" in record:
                        topicid = record["topicid"]
                    elif "pmid" in record:
                        topicid = record["pmid"]
                    else:
                        topicid = ""

                    if record.get("title") and (record.get("references-pmids") or record.get("results-section-pmids")):

                        prompt= build_prompt(record["title"], sample_format=sample_format, tokenizer=tokenizer, enable_thinking=enable_thinking)

                        min_date = record.get("min-date", "")
                        max_date = record.get("max-date", "")
                        first_ref_date = record.get("first_ref_date", "")
                        last_ref_date = record.get("last_ref_date", "")

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


                        # boolean_query = record.get("boolean_query", "")
                        # # Retrieve documents if boolean_query is provided
                        # if boolean_query:
                        #     current_dict["boolean_query"] = boolean_query
                        #     if completion:
                        #         current_dict["completion"] = form_completion(boolean_query, sample_out_format)
                        #     retrieved_docs = record.get("retrieved_ids", [])
                        #     if not retrieved_docs:
                        #         retrieved_docs = cached_retrieve_documents(boolean_query, mindate=min_date, maxdate=max_date)
                        #         print(f"Retrieved {len(retrieved_docs)} documents for query: {boolean_query}")
                        #     current_dict["retrieved_ids"] = retrieved_docs
                        data.append(current_dict)
                except json.JSONDecodeError:
                    continue
    return data

