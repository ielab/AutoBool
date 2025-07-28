from typing import List, Tuple, Dict
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache


FASTAPI_URL = "http://localhost:8000/entrez/query"
MAX_WORKERS = 4

# Robust session with retries -------------------------------------------------
_session = requests.Session()
retry_cfg = Retry(
    total=5, backoff_factor=0.3,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["POST"]
)
_adapter = HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS, max_retries=retry_cfg)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

# -----------------------------------------------------------------------------


@lru_cache(maxsize=1024)
def cached_retrieve_documents(query: str, mindate: str, maxdate: str) -> List[str]:
    """Sends a request to the API and returns a list of document IDs."""
    payload = {"query": query, "mindate": mindate, "maxdate": maxdate}
    if not check_logic(query):
        return []
    try:
        # Increased timeout for potentially long-running queries
        r = _session.post(FASTAPI_URL, json=payload, timeout=200)
        r.raise_for_status()
        data = r.json()
        # Log any errors returned from the API itself
        # if data.get("errors"):
        #     print(f"[API Error] Query '{query[:40]}...': {data['errors']}")
        return data.get("ids", [])
    except requests.exceptions.RequestException as e:
        print(f"[Client Error] Query '{query[:40]}...': {e}")
        return []




# --- Logic Checker ---
def check_logic(bool_query: str) -> bool:
    if not bool_query:
        return False

    # Normalize
    bool_query = bool_query.strip()
    bool_query = re.sub(r'\s+', ' ', bool_query)
    bool_query = re.sub(r'\b(and|or|not)\b', lambda m: m.group(1).upper(), bool_query, flags=re.IGNORECASE)

    # Check balanced parentheses and detect empty ()
    depth = 0
    i = 0
    while i < len(bool_query):
        if bool_query[i] == '(':
            depth += 1
            if i + 1 < len(bool_query) and bool_query[i+1] == ')':
                return False  # Empty parentheses
        elif bool_query[i] == ')':
            depth -= 1
            if depth < 0:
                return False  # Unbalanced
        i += 1
    if depth != 0:
        return False

    # Tokenize
    token_pattern = r'\".*?\"|\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+'
    tokens = re.findall(token_pattern, bool_query, flags=re.IGNORECASE)
    tokens = [t.upper() if t.upper() in {'AND', 'OR', 'NOT'} else t for t in tokens]

    if not tokens:
        return False

    valid_ops = {'AND', 'OR', 'NOT'}

    # Basic sequence logic
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

    if tokens[-1] in valid_ops:
        return False

    return True




def compute_format_reward(content: str) -> Tuple[float, str]:
    if "<think>" in content:
        # Must match both <think> and <answer>, and <think> must have non-empty content
        match = re.search(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*\Z", content, flags=re.DOTALL)
        if not match:
            return -10.0, None
        think_content = match.group(1).strip()
        answer_content = match.group(2).strip()
        if len(think_content) < 1 or len(answer_content) < 1:
            return -10.0, None
        return 10.0, answer_content
    else:
        # Only <answer> required and must be non-empty
        match = re.search(r"<answer>(.*?)</answer>\s*\Z", content, flags=re.DOTALL)
        if not match:
            return -10.0, None
        answer_content = match.group(1).strip()
        if len(answer_content) < 1:
            return -10.0, None
        return 10.0, answer_content


def compute_validity_reward(bool_query: str) -> float:
    return 10 if check_logic(bool_query) else -10



def compute_retrieval_reward(
    retrieved_ids: List[str],
    reference_pmids: List[str],
    grading_config: Dict[str, float] = None
) -> float:
    if grading_config is None:
        grading_config = {
            "max_reward": 20.0,
            "min_reward": -20.0,
            "penalty_no_true_positives": -5.0,
            "precision_penalty_weight": 0.2
        }

    if not retrieved_ids:
        return grading_config["min_reward"], 0.0, 0.0

    retrieved_set = set(retrieved_ids)
    reference_set = set(reference_pmids)
    true_positives = retrieved_set & reference_set

    if not true_positives:
        return grading_config["penalty_no_true_positives"], 0.0, 0.0

    recall = len(true_positives) / len(reference_set) if reference_set else 0
    precision = len(true_positives) / len(retrieved_set) if retrieved_set else 0

    base_reward = recall * grading_config["max_reward"]
    penalty = (1 - precision) * grading_config["precision_penalty_weight"] * grading_config["max_reward"]
    reward = base_reward - penalty

    return max(grading_config["penalty_no_true_positives"], min(grading_config["max_reward"], reward)), recall, precision




def compute_retrieval_reward_v2(
    retrieved_ids: List[str],
    reference_pmids: List[str],
    grading_config: Dict[str, float] = None,
    alpha = 1
) -> Tuple[float, float, float]:
    if grading_config is None:
        grading_config = {
            "max_reward": 20.0,
            "min_reward": -20.0,
            "penalty_no_true_positives": -5.0
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

    # Precision's weight is recall
    reward = 0.5 * grading_config["max_reward"] * (recall + recall**alpha * precision)

    # Bound the reward
    return reward, recall, precision

import math
def compute_retrieval_reward_v3(
    retrieved_ids: List[str],
    reference_pmids: List[str],
    grading_config: Dict[str, float] = None,
    alpha = 1
) -> Tuple[float, float, float]:
    if grading_config is None:
        grading_config = {
            "max_reward": 20.0,
            "min_reward": -20.0,
            "penalty_no_true_positives": -5.0,
            "scale": 100  # or 1000 depending on your typical precision range
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

    # Reward computation
    r = 10 * recall
    scale = grading_config["scale"]
    log_p = math.log(1 + precision * scale) / math.log(1 + scale)
    pr = 10 * recall**alpha * log_p

    reward = r + pr

    return reward, recall, precision



def format_reward_func(completions, **kwargs):
    format_rewards = []
    for completion in completions:
        # if type is list then it
        if isinstance(completion, list):
            content = completion[0]["content"]
        else:
            content = completion

        reward, _ = compute_format_reward(content)
        format_rewards.append(reward)
    return format_rewards

def validity_reward_func(completions, **kwargs):
    validity_rewards = []
    for completion in completions:
        if isinstance(completion, list):
            content = completion[0]["content"]
        else:
            content = completion

        _, bool_query = compute_format_reward(content)
        if bool_query:
            reward = compute_validity_reward(bool_query)
        else:
            reward = -10.0
        validity_rewards.append(reward)
    return validity_rewards


def retrieval_reward_func_v2(completions, ground_truth, first_ref_date, last_ref_date, alpha, **kwargs):
    """
    Calculates retrieval rewards by correctly and concurrently calling the API.
    This version correctly handles the user's input structure and concurrent execution.
    """
    grading_config = kwargs.get("grading_config", None)
    max_workers = kwargs.get("max_workers", 5)  # Defaulting to 5 as in user's original code
    # Step 1: Pre-process all completions to extract content and find valid queries
    tasks_to_run = []
    bool_queries = []  # Keep track of all extracted queries (or None) for final reward calc
    for idx, completion in enumerate(completions):
        if isinstance(completion, list):
            content = completion[0]["content"]
        else:
            content = completion

        _, bq = compute_format_reward(content)
        bool_queries.append(bq)  # Store the query or None

        if bq:
            # If the query is valid, add it to the list of tasks to execute
            tasks_to_run.append({
                "original_index": idx,
                "query": bq,
                "mindate": first_ref_date[idx],
                "maxdate": last_ref_date[idx]
            })

    # Step 2: Retrieve documents in parallel using the corrected concurrent pattern
    retrieved_results = [None] * len(completions)
    if tasks_to_run:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(cached_retrieve_documents, task["query"], task["mindate"], task["maxdate"]): task[
                    "original_index"]
                for task in tasks_to_run
            }

            for future in as_completed(future_to_index):
                original_index = future_to_index[future]
                try:
                    retrieved_ids = future.result()
                    retrieved_results[original_index] = retrieved_ids
                except Exception as e:
                    print(f"[Executor Error] Task for index {original_index} failed: {e}")
                    retrieved_results[original_index] = []

    # Step 3: Calculate rewards based on the retrieval results
    retrieval_rewards = []
    for idx, bq in enumerate(bool_queries):
        if bq is None:
            # The query was invalid from the start
            rr = -20.0
        else:
            current_alpha = alpha[idx]
            rr, _, _ = compute_retrieval_reward_v2(
                retrieved_ids=retrieved_results[idx] or [],  # Use [] if retrieval failed
                reference_pmids=ground_truth[idx],
                grading_config=grading_config,
                alpha=current_alpha
            )
        retrieval_rewards.append(rr)

    return retrieval_rewards

def retrieval_reward_func_v3(completions, ground_truth, first_ref_date, last_ref_date, **kwargs):
    """
    Calculates retrieval rewards by correctly and concurrently calling the API.
    This version correctly handles the user's input structure and concurrent execution.
    """
    grading_config = kwargs.get("grading_config", None)
    max_workers = kwargs.get("max_workers", 5)  # Defaulting to 5 as in user's original code

    # Step 1: Pre-process all completions to extract content and find valid queries
    tasks_to_run = []
    bool_queries = []  # Keep track of all extracted queries (or None) for final reward calc
    for idx, completion in enumerate(completions):
        if isinstance(completion, list):
            content = completion[0]["content"]
        else:
            content = completion

        _, bq = compute_format_reward(content)
        bool_queries.append(bq)  # Store the query or None

        if bq:
            # If the query is valid, add it to the list of tasks to execute
            tasks_to_run.append({
                "original_index": idx,
                "query": bq,
                "mindate": first_ref_date[idx],
                "maxdate": last_ref_date[idx]
            })

    # Step 2: Retrieve documents in parallel using the corrected concurrent pattern
    retrieved_results = [None] * len(completions)
    if tasks_to_run:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(cached_retrieve_documents, task["query"], task["mindate"], task["maxdate"]): task[
                    "original_index"]
                for task in tasks_to_run
            }

            for future in as_completed(future_to_index):
                original_index = future_to_index[future]
                try:
                    retrieved_ids = future.result()
                    retrieved_results[original_index] = retrieved_ids
                except Exception as e:
                    print(f"[Executor Error] Task for index {original_index} failed: {e}")
                    retrieved_results[original_index] = []

    # Step 3: Calculate rewards based on the retrieval results
    retrieval_rewards = []
    for idx, bq in enumerate(bool_queries):
        if bq is None:
            # The query was invalid from the start
            rr = -20.0
        else:
            rr, _, _ = compute_retrieval_reward_v3(
                retrieved_ids=retrieved_results[idx] or [],  # Use [] if retrieval failed
                reference_pmids=ground_truth[idx],
                grading_config=grading_config,
            )
        retrieval_rewards.append(rr)

    return retrieval_rewards

# --- Reward Function ---
def reward_func(completions, ground_truth, first_ref_date, last_ref_date, **kwargs):
    grading_config = kwargs.get("grading_config", None)

    # Extract completion contents
    completion_contents = [completion[0]["content"] for completion in completions]

    # Precompute format reward and extract valid queries
    bool_queries = []
    valid_indices = []
    date_pairs = []
    format_rewards = [0.0] * len(completion_contents)

    for idx, content in enumerate(completion_contents):
        fr, bq = compute_format_reward(content)
        format_rewards[idx] = fr
        if bq:
            bool_queries.append(bq)
            valid_indices.append(idx)
            date_pairs.append((first_ref_date[idx], last_ref_date[idx]))
        else:
            bool_queries.append(None)  # Keep alignment

    # Retrieve documents in parallel
    retrieved_results = [None] * len(completion_contents)
    if valid_indices:
        with ThreadPoolExecutor(max_workers=kwargs.get("max_workers", 5)) as executor:
            futures = {
                executor.submit(cached_retrieve_documents, bool_queries[i], *date_pairs[j]): valid_indices[j]
                for j, i in enumerate(valid_indices)
            }
            for future in futures:
                idx = futures[future]
                try:
                    _, retrieved_ids = future.result()
                except Exception:
                    retrieved_ids = []
                retrieved_results[idx] = retrieved_ids

    # Compute rewards
    recalls = []
    precisions = []
    validity_rewards = []
    retrieval_rewards = []
    rewards = []

    for idx, bq in enumerate(bool_queries):
        fr = format_rewards[idx]
        if bq is None:
            # Penalty for invalid query
            vr = -10.0
            rr = -20.0
            recall = 0.0
            precision = 0.0
        else:
            vr = compute_validity_reward(bq)
            rr, recall, precision = compute_retrieval_reward(
                retrieved_ids=retrieved_results[idx] or [],
                reference_pmids=ground_truth[idx],
                grading_config=grading_config,
            )
        rewards.append(fr + vr + rr)
        recalls.append(recall)
        precisions.append(precision)
        validity_rewards.append(vr)
        retrieval_rewards.append(rr)

    # Log averages

    return rewards

if __name__ == "__main__":
    completions = [
        [{"role": "assistant", "content": '<think>Heart disease + diabetes</think><answer>"heart diseases"[MeSH Terms] AND "diabetes mellitus"[MeSH Terms]</answer>'}],
        [{"role": "assistant", "content": '<think>Aspirin trials in elderly</think><answer>(("aspirin"[MeSH Terms] AND "myocardial infarction"[MeSH Terms]) AND ("aged"[MeSH Terms] OR "elderly"[Title/Abstract]) AND "randomized controlled trial"[Publication Type] AND humans[MeSH Terms]) NOT ("animals"[MeSH Terms] NOT "humans"[MeSH Terms] OR "gastrointestinal hemorrhage"[MeSH Terms] OR bleeding[Title/Abstract])</answer>'}],
        [{"role": "assistant", "content": '<think>COVID-19 and vaccines</think><answer>"COVID-19"[MeSH Terms] AND "vaccines"[MeSH Terms]</answer>'}],
        [{"role": "assistant", "content": '<think>Alzheimer\'s and cognitive decline</think><answer>"Alzheimer Disease"[MeSH Terms] AND "Cognition Disorders"[MeSH Terms]</answer>'}],
        [{"role": "assistant", "content": '<think>Obesity and hypertension</think><answer>"Obesity"[MeSH Terms] AND "Hypertension"[MeSH Terms]</answer>'}],
        [{"role": "assistant", "content": '<think>Depression treatments in adolescents</think><answer>"Depressive Disorder"[MeSH Terms] AND "Adolescent"[MeSH Terms] AND "Therapy"[Subheading]</answer>'}],
        [{"role": "assistant", "content": '<think>Cancer immunotherapy</think><answer>"Neoplasms"[MeSH Terms] AND "Immunotherapy"[MeSH Terms]</answer>'}],
        [{"role": "assistant", "content": '<think>Diabetes prevention lifestyle</think><answer>"Diabetes Mellitus, Type 2"[MeSH Terms] AND ("Exercise"[MeSH Terms] OR "Diet"[MeSH Terms])</answer>'}],
        [{"role": "assistant", "content": '<think>Smoking and lung cancer</think><answer>"Smoking"[MeSH Terms] AND "Lung Neoplasms"[MeSH Terms]</answer>'}],
        [{"role": "assistant", "content": '<think>COVID-19 + telemedicine</think><answer>"COVID-19"[MeSH Terms] AND "Telemedicine"[MeSH Terms]</answer>'}],
        [{"role": "assistant", "content": '<think>Stroke rehabilitation</think><answer>"Stroke"[MeSH Terms] AND "Rehabilitation"[MeSH Terms]</answer>'}],
        [{"role": "assistant", "content": '<think>Childhood vaccination and autism</think><answer>"Vaccination"[MeSH Terms] AND "Autism Spectrum Disorder"[MeSH Terms]</answer>'}],
        [{"role": "assistant", "content": '<think>Heart failure drugs</think><answer>"Heart Failure"[MeSH Terms] AND "Drug Therapy"[Subheading]</answer>'}],
        [{"role": "assistant", "content": '<think>Antibiotic resistance in E. coli</think><answer>"Escherichia coli"[MeSH Terms] AND "Drug Resistance, Bacterial"[MeSH Terms]</answer>'}],
        [{"role": "assistant", "content": '<think>Mental health and sleep</think><answer>"Mental Health"[MeSH Terms] AND "Sleep"[MeSH Terms]</answer>'}],
    ]

    # Dummy but structurally valid PMIDs for ground truth
    ground_truth = [
        ["10921480", "9927831", "3"],
        ["32795098", "37345815"],
        ["32112345", "32156789"],
        ["28976543", "29000123"],
        ["30123456", "30123457"],
        ["31098765", "31098766"],
        ["31876543", "31876544"],
        ["33333333", "33333334"],
        ["29999999", "30000001"],
        ["34343434", "34343435"],
        ["27888888", "27888889"],
        ["26543210", "26543211"],
        ["25000000", "25000001"],
        ["28888888", "28888889"],
        ["21212121", "21212122"],
    ]

    # Reasonable reference date windows
    first_ref_date = [
        "1999/01/01", "2023/10/01", "2020/01/01", "2018/01/01", "2015/01/01",
        "2010/01/01", "2019/06/01", "2012/01/01", "2008/01/01", "2021/01/01",
        "2011/01/01", "2005/01/01", "2000/01/01", "2016/01/01", "2013/01/01"
    ]
    last_ref_date = [
        "2000/10/01", "2023/10/31", "2021/01/01", "2019/01/01", "2016/01/01",
        "2011/01/01", "2020/06/01", "2013/01/01", "2009/01/01", "2022/01/01",
        "2012/01/01", "2006/01/01", "2001/01/01", "2017/01/01", "2014/01/01"
    ]

    # Run with a fast 8-thread setup
    from time import time
    start = time()
    rewards = reward_func(
        completions,
        ground_truth=ground_truth,
        first_ref_date=first_ref_date,
        last_ref_date=last_ref_date,
    )
    print("Rewards:", rewards)
    print(f"Time taken for 15 examples: {time() - start:.2f}s")

