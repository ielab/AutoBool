from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Dict, Tuple
from Bio import Entrez
import pandas as pd
import time

# === CONFIGURATION ===
email = "xxx@gmail.com"
time_max = 10000
overall_max = 200000
max_attempt_count = 10

Entrez.email = email
app = FastAPI(title="PubMed Query API")

# === INPUT SCHEMA ===
class QueryRequest(BaseModel):
    query: str
    mindate: str
    maxdate: str

# === UTIL FUNCTIONS ===
def dates_check(date: str) -> str:
    if "/" not in date:
        return f"{date[:4]}/{date[4:6]}/{date[6:]}"
    parts = date.split("/")
    return date if len(parts[0]) == 4 else f"{parts[2]}/{parts[1]}/{parts[0]}"

def submission_one(query: str, current_min_date: str, current_max_date: str) -> Tuple[int, List[str]]:
    attempt_count = 0
    while attempt_count < max_attempt_count:
        try:
            handle = Entrez.esearch(
                db="pubmed", term=query, retmax=time_max,
                mindate=current_min_date, maxdate=current_max_date
            )
            record = Entrez.read(handle)
            return int(record["Count"]), record["IdList"]
        except Exception as e:
            print(f"Error: {e}")
            attempt_count += 1
            time.sleep(1)
    return 0, []

def pubmed_submission(query: str, dates: Dict[str, str], counter_too_many: int = 0) -> Tuple[List[str], int]:
    original_chunks = [(dates_check(dates["mindate"]), dates_check(dates["maxdate"]))]
    id_lists = []
    while original_chunks:
        start, end = original_chunks.pop(0)
        count, ids = submission_one(query, start, end)
        if count > overall_max:
            counter_too_many += 1
            break
        if count > time_max:
            ts1, ts2 = pd.Timestamp(start), pd.Timestamp(end)
            delta = (ts2 - ts1) / 3
            prev = start
            for i in range(1, 3):
                mid = str(ts1 + i * delta)[:10]
                original_chunks.append((prev, mid))
                prev = mid
            original_chunks.append((prev, end))
        else:
            id_lists.extend(ids)
    return list(set(id_lists)), counter_too_many

# === ENDPOINT (Concurrent-Safe) ===
@app.post("/entrez/query")
async def query_pubmed(request: QueryRequest):
    result = await run_in_threadpool(pubmed_submission, request.query, {
        "mindate": request.mindate,
        "maxdate": request.maxdate
    }, 0)
    id_list, too_many_count = result
    return {
        "query": request.query,
        "total_retrieved": len(id_list),
        "too_many_chunks": too_many_count,
        "ids": id_list
    }