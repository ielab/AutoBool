import time
import random
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import lru_cache

# === CONFIGURATION ===
# The URL of your running FastAPI application
FASTAPI_URL = "http://localhost:8000/entrez/query"

# Number of concurrent requests to send to the API
# Simulating 30 jobs with 5 threads each (30 * 5 = 150)
MAX_WORKERS = 150

# === TEST DATA ===
# A list of query payloads to send.
# We include many duplicates to simulate popular queries and test caching.
TEST_QUERIES = [
    # A standard query that should be under the RETMAX limit
    {
        "query": "cancer[ti] AND blood[tiab]",
        "mindate": "2020/01/01",
        "maxdate": "2022/12/31"
    },
    # A very large query that will trigger the splitting logic
    {
        "query": "cancer[ti]",
        "mindate": "2021/01/01",
        "maxdate": "2021/12/31"
    },
    # A query with no results
    {
        "query": "asdfghjklqwerty[tiab]",
        "mindate": "2000/01/01",
        "maxdate": "2022/12/31"
    },
]

# Create a much larger list of queries for the heavier stress test,
# simulating multiple jobs making repeated, cacheable requests.
STRESS_TEST_PAYLOADS = (
    [TEST_QUERIES[0]] * 20 +  # Simulates 15 jobs making 5 requests for the standard query
    [TEST_QUERIES[1]] * 20 +  # Simulates 12 jobs making 5 requests for the large query
    [TEST_QUERIES[2]] * 10     # Simulates 3 jobs making 5 requests for the no-results query
)
random.shuffle(STRESS_TEST_PAYLOADS)


# === ROBUST HTTP SESSION SETUP ===
# A single session for all threads, which is thread-safe and efficient.
_session = requests.Session()
# Configure retry strategy for robustness against temporary server errors
retry_cfg = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["POST"]
)
_adapter = HTTPAdapter(max_retries=retry_cfg)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


# === API REQUEST FUNCTION ===
# The lru_cache is applied here. Note: to truly load-test the API without caching,
# you can comment out the "@lru_cache" decorator.
@lru_cache(maxsize=1024)
def retrieve_documents(payload_tuple: Tuple[Tuple[str, Any], ...]) -> Dict[str, Any]:
    """
    Sends a single request to the FastAPI endpoint and returns the JSON response.
    The payload is passed as a hashable tuple for caching.
    """
    # Convert the hashable tuple back into a dictionary for use in the request
    payload = dict(payload_tuple)
    query = payload.get("query", "unknown")
    print(f"[Submitting] Query: '{query[:30]}...'")

    try:
        # Increased timeout for potentially long-running queries
        r = _session.post(FASTAPI_URL, json=payload, timeout=300)
        # Raise an exception for bad status codes (4xx or 5xx)
        r.raise_for_status()

        # Return the full JSON response
        response_data = r.json()
        response_data["_success"] = True
        return response_data

    except requests.exceptions.RequestException as e:
        print(f"[Error] Query '{query[:30]}...' failed: {e}")
        return {"query": query, "errors": [str(e)], "_success": False}


def main():
    """
    Runs the high-concurrency test against the FastAPI service.
    """
    print(f"🚀 Starting high-concurrency test with {MAX_WORKERS} workers...")
    print(f"   Sending a total of {len(STRESS_TEST_PAYLOADS)} requests to {FASTAPI_URL}")

    start_time = time.time()
    results = []
    success_count = 0
    failure_count = 0

    # Using ThreadPoolExecutor to send requests concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a dictionary to map future objects back to their original payload for logging
        future_to_payload = {}
        for payload in STRESS_TEST_PAYLOADS:
            # Convert the dict to a sorted tuple of items to make it hashable for lru_cache
            payload_tuple = tuple(sorted(payload.items()))
            future = executor.submit(retrieve_documents, payload_tuple)
            future_to_payload[future] = payload # Map the future to the original dict

        for future in as_completed(future_to_payload):
            payload = future_to_payload[future]
            try:
                result = future.result()
                results.append(result)

                if result.get("_success"):
                    success_count += 1
                    print(f"[Result OK] Query '{result.get('query', 'N/A')[:30]}...' returned {result.get('total_retrieved', 'N/A')} IDs.")
                else:
                    failure_count += 1
                    print(f"[Result FAIL] Query '{payload.get('query', 'N/A')[:30]}...' failed. Error: {result.get('errors')}")

            except Exception as exc:
                failure_count += 1
                print(f"[FATAL] Query '{payload.get('query', 'N/A')[:30]}...' generated an exception: {exc}")
                results.append({"query": payload.get('query'), "errors": [str(exc)], "_success": False})

    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Total requests sent: {len(STRESS_TEST_PAYLOADS)}")
    print(f"Successful requests: {success_count}")
    print(f"Failed requests: {failure_count}")
    if success_count > 0:
        print(f"Average time per successful request: {total_time / success_count:.2f} seconds")
    print("="*50)


if __name__ == "__main__":
    main()
