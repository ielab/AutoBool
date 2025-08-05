"""
Reward functions for GRPO training with Boolean query validation and retrieval evaluation.

This module provides functions to evaluate the quality of generated Boolean queries
for medical literature search, including format validation, logic checking, and
retrieval performance assessment.
"""

import math
import re
from typing import List, Tuple, Dict, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from logging_config import get_logger

# Setup logger for reward functions
logger = get_logger("autobool.reward")

# Configuration
FASTAPI_URL = "http://localhost:8000/entrez/query"
MAX_WORKERS = 4


# Session setup with retry logic
def _create_session() -> requests.Session:
    """Create a robust HTTP session with retry configuration."""
    session = requests.Session()
    retry_config = Retry(
        total=5,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
        max_retries=retry_config
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_session = _create_session()


# Core validation functions
def check_logic(bool_query: str) -> bool:
    """
    Validate the logical structure of a Boolean query.

    Args:
        bool_query: The Boolean query string to validate

    Returns:
        True if the query has valid logical structure, False otherwise
    """
    if not bool_query or not bool_query.strip():
        return False

    # Normalize query
    query = bool_query.strip()
    query = re.sub(r'\s+', ' ', query)
    query = re.sub(r'\b(and|or|not)\b', lambda m: m.group(1).upper(), query, flags=re.IGNORECASE)

    # Check balanced parentheses and detect empty ()
    depth = 0
    for i, char in enumerate(query):
        if char == '(':
            depth += 1
            if i + 1 < len(query) and query[i + 1] == ')':
                return False  # Empty parentheses
        elif char == ')':
            depth -= 1
            if depth < 0:
                return False  # Unbalanced

    if depth != 0:
        return False

    # Tokenize and validate sequence
    token_pattern = r'\".*?\"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+'
    tokens = re.findall(token_pattern, query, flags=re.IGNORECASE)
    tokens = [t.upper() if t.upper() in {'AND', 'OR', 'NOT'} else t for t in tokens]

    if not tokens:
        return False

    # Validate token sequence
    valid_ops = {'AND', 'OR', 'NOT'}
    prev = None

    for i, token in enumerate(tokens):
        if token in {'AND', 'OR'}:
            if prev is None or prev in valid_ops or prev == '(':
                return False
        elif token == 'NOT':
            if i == len(tokens) - 1:
                return False
            if tokens[i + 1] in valid_ops or tokens[i + 1] == ')':
                return False
        elif token == '(':
            if prev and prev not in valid_ops and prev != '(':
                return False
        elif token == ')':
            if prev in valid_ops or prev == '(':
                return False
        prev = token

    return tokens[-1] not in valid_ops


def compute_format_reward(content: str) -> Tuple[float, Optional[str]]:
    """
    Evaluate the format of model completion and extract query.

    Args:
        content: The model's completion content

    Returns:
        Tuple of (reward, extracted_query). Query is None if format is invalid.
    """
    if "<think>" in content:
        # Format with thinking and answer tags
        match = re.search(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*\Z",
                          content, flags=re.DOTALL)
        if not match:
            return -10.0, None

        think_content = match.group(1).strip()
        answer_content = match.group(2).strip()

        if not think_content or not answer_content:
            return -10.0, None
        return 10.0, answer_content
    else:
        # Format with only answer tag
        match = re.search(r"<answer>(.*?)</answer>\s*\Z", content, flags=re.DOTALL)
        if not match:
            return -10.0, None

        answer_content = match.group(1).strip()
        if not answer_content:
            return -10.0, None
        return 10.0, answer_content


# Document retrieval
@lru_cache(maxsize=1024)
def cached_retrieve_documents(query: str, mindate: str, maxdate: str) -> List[str]:
    """
    Retrieve document IDs from the API with caching.

    Args:
        query: Boolean query string
        mindate: Minimum publication date
        maxdate: Maximum publication date

    Returns:
        List of document IDs
    """
    if not check_logic(query):
        return []

    payload = {"query": query, "mindate": mindate, "maxdate": maxdate}

    try:
        response = _session.post(FASTAPI_URL, json=payload, timeout=200)
        response.raise_for_status()
        data = response.json()
        return data.get("ids", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Client Error - Query '{query[:40]}...': {e}")
        return []


# Reward computation functions
def compute_validity_reward(bool_query: str) -> float:
    """Compute reward based on query validity."""
    return 10.0 if check_logic(bool_query) else -10.0


def compute_retrieval_reward(
        retrieved_ids: List[str],
        reference_pmids: List[str],
        grading_config: Optional[Dict[str, float]] = None,
        alpha: float = 1.0
) -> Tuple[float, float, float]:
    """
    Compute retrieval reward based on precision and recall with logarithmic scaling.

    Args:
        retrieved_ids: List of retrieved document IDs
        reference_pmids: List of ground truth document IDs
        grading_config: Configuration for reward calculation
        alpha: Exponent for recall weighting

    Returns:
        Tuple of (reward, recall, precision)
    """
    if grading_config is None:
        grading_config = {
            "max_reward": 20.0,
            "min_reward": -20.0,
            "penalty_no_true_positives": -5.0,
            "scale": 100
        }

    if not retrieved_ids:
        return grading_config["min_reward"], 0.0, 0.0

    retrieved_set = set(retrieved_ids)
    reference_set = set(reference_pmids)
    true_positives = retrieved_set & reference_set

    if not true_positives:
        return grading_config["penalty_no_true_positives"], 0.0, 0.0

    recall = len(true_positives) / len(reference_set) if reference_set else 0.0
    precision = len(true_positives) / len(retrieved_set) if retrieved_set else 0.0

    # Compute reward with logarithmic precision scaling
    recall_reward = 10 * recall
    scale = grading_config["scale"]
    log_precision = math.log(1 + precision * scale) / math.log(1 + scale)
    precision_reward = 10 * (recall ** alpha) * log_precision

    total_reward = recall_reward + precision_reward
    return total_reward, recall, precision


# Main reward functions for training
def format_reward_func(completions, **kwargs) -> List[float]:
    """Compute format rewards for a batch of completions."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        reward, _ = compute_format_reward(content)
        rewards.append(reward)
    return rewards


def validity_reward_func(completions, **kwargs) -> List[float]:
    """Compute validity rewards for a batch of completions."""
    rewards = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        _, query = compute_format_reward(content)
        reward = compute_validity_reward(query) if query else -10.0
        rewards.append(reward)
    return rewards


def retrieval_reward_func(
        completions,
        ground_truth,
        first_ref_date,
        last_ref_date,
        alpha=None,
        **kwargs
) -> List[float]:
    """
    Compute retrieval rewards for a batch of completions with concurrent API calls.

    Args:
        completions: List of model completions
        ground_truth: List of reference PMID lists
        first_ref_date: List of minimum dates
        last_ref_date: List of maximum dates
        alpha: List of alpha values for recall weighting
        **kwargs: Additional configuration

    Returns:
        List of retrieval rewards
    """
    grading_config = kwargs.get("grading_config")
    max_workers = kwargs.get("max_workers", 5)

    # Extract queries and prepare tasks
    tasks_to_run = []
    bool_queries = []

    for idx, completion in enumerate(completions):
        content = completion[0]["content"] if isinstance(completion, list) else completion
        _, query = compute_format_reward(content)
        bool_queries.append(query)

        if query:
            tasks_to_run.append({
                "original_index": idx,
                "query": query,
                "mindate": first_ref_date[idx],
                "maxdate": last_ref_date[idx]
            })

    # Execute retrieval tasks concurrently
    retrieved_results = [None] * len(completions)
    if tasks_to_run:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    cached_retrieve_documents,
                    task["query"],
                    task["mindate"],
                    task["maxdate"]
                ): task["original_index"]
                for task in tasks_to_run
            }

            for future in as_completed(future_to_index):
                original_index = future_to_index[future]
                try:
                    retrieved_ids = future.result()
                    retrieved_results[original_index] = retrieved_ids
                except Exception as e:
                    logger.error(f"Executor Error - Task for index {original_index} failed: {e}")
                    retrieved_results[original_index] = []

    # Calculate rewards
    rewards = []
    for idx, query in enumerate(bool_queries):
        if query is None:
            rewards.append(-20.0)
        else:
            current_alpha = alpha[idx] if alpha and isinstance(alpha, list) else (alpha or 1.0)
            reward, _, _ = compute_retrieval_reward(
                retrieved_ids=retrieved_results[idx] or [],
                reference_pmids=ground_truth[idx],
                grading_config=grading_config,
                alpha=current_alpha
            )
            rewards.append(reward)

    return rewards


# Example usage and testing
if __name__ == "__main__":
    # Test data
    completions = [
        [{"role": "assistant",
          "content": '<think>Heart disease + diabetes</think><answer>"heart diseases"[MeSH Terms] AND "diabetes mellitus"[MeSH Terms]</answer>'}],
        [{"role": "assistant",
          "content": '<think>COVID-19 and vaccines</think><answer>"COVID-19"[MeSH Terms] AND "vaccines"[MeSH Terms]</answer>'}],
    ]

    ground_truth = [
        ["10921480", "9927831"],
        ["32112345", "32156789"],
    ]

    first_ref_date = ["1999/01/01", "2020/01/01"]
    last_ref_date = ["2000/10/01", "2021/01/01"]

    # Test reward functions
    format_rewards = format_reward_func(completions)
    validity_rewards = validity_reward_func(completions)
    retrieval_rewards = retrieval_reward_func(
        completions, ground_truth, first_ref_date, last_ref_date
    )

    logger.info(f"Format rewards: {format_rewards}")
    logger.info(f"Validity rewards: {validity_rewards}")
    logger.info(f"Retrieval rewards: {retrieval_rewards}")