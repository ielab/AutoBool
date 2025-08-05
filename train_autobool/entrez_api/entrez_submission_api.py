import time
import random
import asyncio
import logging
from typing import List, Dict, Tuple, Optional

# Third-party libraries
import pandas as pd
import aiohttp
from lxml import etree
from fastapi import FastAPI
from pydantic import BaseModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("entrez_api")

# === CONFIGURATION ===
API_KEYS = [
    
]
RETMAX = 10_000
OVERALL_MAX = 200_000
CONCURRENT_REQUESTS = 4
ENTREZ_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
KEY_COOLDOWN_SECONDS = 3
WORKER_BACKOFF_SECONDS = 5
# New resiliency settings
MAX_RETRIES_PER_CHUNK = 5
RETRY_BACKOFF_FACTOR_SECONDS = 2

# Create the FastAPI application
app = FastAPI(title="Resilient & Concurrent PubMed Query API")


# === I/O DATA MODEL ===
class QueryRequest(BaseModel):
    query: str
    mindate: str
    maxdate: str


# === STATEFUL API KEY POOL ===
class APIPool:
    """Manages the state of API keys for a single user request."""

    def __init__(self):
        self._keys = API_KEYS.copy()
        self._cooldown = {}
        self._lock = asyncio.Lock()

    async def get_key(self) -> Optional[Dict[str, str]]:
        """Get an available key that is not on cooldown."""
        async with self._lock:
            random.shuffle(self._keys)
            now = time.monotonic()
            for key in self._keys:
                if key["email"] not in self._cooldown or now > self._cooldown[key["email"]]:
                    return key
            return None

    async def report_failure(self, key_email: str):
        """Put a key on cooldown after a failure."""
        async with self._lock:
            cooldown_end = time.monotonic() + KEY_COOLDOWN_SECONDS
            self._cooldown[key_email] = cooldown_end
            logger.warning(f"Key '{key_email}' on cooldown for {KEY_COOLDOWN_SECONDS}s")


# === ASYNC HELPER FUNCTIONS ===
def dates_check(date: str) -> str:
    # ... (unchanged)
    if "/" in date:
        parts = date.split("/")
        return f"{parts[2]}/{parts[1]}/{parts[0]}" if len(parts[0]) != 4 else date
    elif len(date) == 8:
        return f"{date[:4]}/{date[4:6]}/{date[6:]}"
    return date


async def fetch_and_count_async(
        session: aiohttp.ClientSession, api_pool: APIPool, key: Dict[str, str], query: str, start: str, end: str
) -> Tuple[List[str], int, Optional[str]]:
    """Makes a single API call with a specific key."""
    params = {
        "db": "pubmed", "term": query, "mindate": start, "maxdate": end,
        "retmax": RETMAX, "usehistory": "y", "email": key["email"],
        "api_key": key["key"], "retmode": "xml"
    }
    try:
        async with session.get(ENTREZ_URL, params=params, timeout=120) as response:
            if response.status == 429:
                await api_pool.report_failure(key["email"])
                error_message = f"Rate limit hit (429) for key '{key['email']}'"
                return [], -1, error_message

            response.raise_for_status()
            content = await response.read()
            root = etree.fromstring(content)
            count = int(root.findtext("Count", "0"))
            ids = [id_tag.text for id_tag in root.xpath("//IdList/Id")]
            return ids, count, None

    except Exception as e:
        error_message = f"API call failed: {e}"
        return [], -1, error_message


# === ASYNC CORE LOGIC ===
async def pubmed_split_by_date_async(query: str, mindate: str, maxdate: str) -> Tuple[List[str], int, List[str]]:
    all_ids = set()
    error_log = []
    api_pool = APIPool()
    mindate, maxdate = dates_check(mindate), dates_check(maxdate)

    # Initial quick check to see total count before starting full process
    initial_key = await api_pool.get_key()
    if not initial_key:
        error_log.append("No available API key for initial check.")
        return [], 1, error_log

    async with aiohttp.ClientSession() as session:
        ids, count, error = await fetch_and_count_async(session, api_pool, initial_key, query, mindate, maxdate)

        if error:
            error_log.append(f"Initial count check failed: {error}")
            return [], 1, error_log

        if count > OVERALL_MAX:
            error_message = f"Total count {count} exceeds OVERALL_MAX ({OVERALL_MAX}). Stopping early."
            error_log.append(error_message)
            logger.warning(error_message)
            return [], 1, error_log

        # If the count is acceptable, start processing
        work_queue = asyncio.Queue()
        work_queue.put_nowait((mindate, maxdate))

        async def worker(session: aiohttp.ClientSession):
            while True:
                start, end = await work_queue.get()

                attempts = 0
                while attempts < MAX_RETRIES_PER_CHUNK:
                    key = await api_pool.get_key()
                    if not key:
                        await asyncio.sleep(WORKER_BACKOFF_SECONDS)
                        continue

                    ids, count, error = await fetch_and_count_async(session, api_pool, key, query, start, end)


                    if not error:
                        if count > 0 and ids:
                            all_ids.update(ids)

                        if count > RETMAX:
                            start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
                            if (duration := end_ts - start_ts).days > 0:
                                data_ratio = RETMAX / count
                                time_split_ratio = 0.9 - (0.4 * data_ratio)
                                split_point_ts = start_ts + duration * time_split_ratio
                                chunk1_end = split_point_ts.strftime('%Y/%m/%d')
                                chunk2_start = (split_point_ts + pd.Timedelta(days=1)).strftime('%Y/%m/%d')

                                if pd.Timestamp(chunk2_start) <= end_ts:
                                    work_queue.put_nowait((chunk2_start, end))
                                if start_ts <= pd.Timestamp(chunk1_end):
                                    work_queue.put_nowait((start, chunk1_end))

                        break

                    attempts += 1
                    if attempts < MAX_RETRIES_PER_CHUNK:
                        delay = RETRY_BACKOFF_FACTOR_SECONDS * (2 ** (attempts - 1))
                        await asyncio.sleep(delay)
                    else:
                        final_error = f"FATAL: Chunk '{start}-{end}' failed after {MAX_RETRIES_PER_CHUNK} attempts. Last error: {error}"
                        error_log.append(final_error)

                work_queue.task_done()

        workers = [asyncio.create_task(worker(session)) for _ in range(CONCURRENT_REQUESTS)]
        await work_queue.join()
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    total_retrieved = len(all_ids)
    exceeded_flag = 1 if total_retrieved > OVERALL_MAX else 0
    return list(all_ids), exceeded_flag, error_log


# === ASYNC CORE LOGIC ===
async def pubmed_split_by_date_async(query: str, mindate: str, maxdate: str) -> Tuple[List[str], int, List[str]]:
    all_ids = set()
    error_log = []
    api_pool = APIPool()
    mindate, maxdate = dates_check(mindate), dates_check(maxdate)

    work_queue = asyncio.Queue()
    await work_queue.put((mindate, maxdate))

    exceeded_flag = 0
    exceeded_event = asyncio.Event()  # Async-safe flag

    async def worker(session: aiohttp.ClientSession):
        nonlocal exceeded_flag

        while not exceeded_event.is_set():
            try:
                start, end = await asyncio.wait_for(work_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue  # Allows check for exceeded_event even if queue is idle

            attempts = 0
            while attempts < MAX_RETRIES_PER_CHUNK:
                key = await api_pool.get_key()
                if not key:
                    logger.warning(f"All keys on cooldown. Sleeping for {WORKER_BACKOFF_SECONDS}s...")
                    await asyncio.sleep(WORKER_BACKOFF_SECONDS)
                    continue

                ids, count, error = await fetch_and_count_async(session, api_pool, key, query, start, end)

                if not error:
                    if count > OVERALL_MAX:
                        logger.error(f"Chunk '{start}-{end}' has {count} results, exceeding OVERALL_MAX={OVERALL_MAX}")
                        exceeded_flag = 1
                        exceeded_event.set()
                        work_queue.task_done()
                        return

                    if count > 0 and ids:
                        all_ids.update(ids)
                    logger.info(f"Retrieved {len(ids)}/{count} IDs from '{start}-{end}'. Total: {len(all_ids)}")

                    if count > RETMAX:
                        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
                        if (duration := end_ts - start_ts).days > 0:
                            data_ratio = RETMAX / count
                            time_split_ratio = 0.9 - (0.4 * data_ratio)
                            split_point_ts = start_ts + duration * time_split_ratio
                            chunk1_end = split_point_ts.strftime('%Y/%m/%d')
                            chunk2_start = (split_point_ts + pd.Timedelta(days=1)).strftime('%Y/%m/%d')

                            if pd.Timestamp(chunk2_start) <= end_ts:
                                await work_queue.put((chunk2_start, end))
                            if start_ts <= pd.Timestamp(chunk1_end):
                                await work_queue.put((start, chunk1_end))

                    break  # success

                # On error
                attempts += 1
                if attempts < MAX_RETRIES_PER_CHUNK:
                    delay = RETRY_BACKOFF_FACTOR_SECONDS * (2 ** (attempts - 1))
                    logger.warning(f"Chunk '{start}-{end}' failed (attempt {attempts}/{MAX_RETRIES_PER_CHUNK}). Retrying in {delay}s. Error: {error}")
                    await asyncio.sleep(delay)
                else:
                    final_error = f"FATAL: Chunk '{start}-{end}' failed after {MAX_RETRIES_PER_CHUNK} attempts. Last error: {error}"
                    logger.error(final_error)
                    error_log.append(final_error)

            work_queue.task_done()

    async with aiohttp.ClientSession() as session:
        workers = [asyncio.create_task(worker(session)) for _ in range(CONCURRENT_REQUESTS)]
        await work_queue.join()
        exceeded_event.set()  # stop workers in case they are waiting on queue
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    logger.info(f"Query complete. Unique IDs: {len(all_ids)}, Exceeded limit: {exceeded_flag}")
    return list(all_ids), exceeded_flag, error_log


@app.post("/entrez/query")
async def query_pubmed(request: QueryRequest):
    """Accepts a query, start date, and end date."""
    id_list, exceeded_flag, errors = await pubmed_split_by_date_async(
        request.query, request.mindate, request.maxdate)
    return {
        "query": request.query, "date_range": f"{request.mindate} to {request.maxdate}",
        "total_retrieved": len(id_list), "exceeded_overall_max": bool(exceeded_flag),
        "ids": id_list, "errors": errors
    }