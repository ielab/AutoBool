#!/usr/bin/env python3
"""
Iterative, resumable GPT-4o Batch-API evaluation.

Fixes / improvements
--------------------
* Robust download via `client.files.content(id).read()`
* Resilient `parse_output()` (handles always-present `"error": null`)
* Retrieval list initialised correctly
* NEW helper `write_eval_summary()` adds one AVG line after all rounds
* ✅ Retrieval dates controlled by dataset and --ref_date
* FIXED: Only process successful queries, track solved topics properly
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple
from tqdm import tqdm  # add this at the top of your script
import numpy as np
from datasets import load_dataset, Dataset
from openai import OpenAI

client = OpenAI()


###############################################################################
# Utility helpers
###############################################################################

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def extract_boolean_query(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def compute_recall_precision(ret: List[str], ref: List[str]) -> Tuple[float, float]:
    if not ret or not ref:
        return 0.0, 0.0
    tp = set(ret) & set(ref)
    return len(tp) / len(ref), len(tp) / len(ret)


###############################################################################
# Batch helpers
###############################################################################

def make_batch_input(
        rows: List[Dict],
        model: str,
        temp: float,
        max_prompt_length: int,
        max_completion_length: int,
        out_path: Path,
):
    with out_path.open("w", encoding="utf-8") as fp:
        for r in rows:
            if "o3" in model or "o1" in model:
                current_dict = {
                            "custom_id": str(r["topicid"]),
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": model,
                                "messages": r["prompt"],
                                "max_completion_tokens": max_completion_length,
                            },
                        }
            else:
                current_dict = {
                            "custom_id": str(r["topicid"]),
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": model,
                                "messages": r["prompt"],
                                "max_tokens": max_completion_length+ max_prompt_length,
                                "temperature": temp,
                                "top_p": 0.95,
                            },
                        }

            fp.write(
                json.dumps(current_dict,
                    ensure_ascii=False,
                )
                + "\n"
            )


def submit_batch(jsonl_path: Path) -> str:
    file_obj = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


def wait_and_download(bid: str, dest: Path, poll: int):
    print(f"[WAIT] {bid} – polling every {poll}s")
    while True:
        b = client.batches.retrieve(bid)
        if b.status in {"failed", "expired", "cancelled"}:
            raise RuntimeError(f"Batch {bid} ended with status {b.status}")
        if b.status == "completed":
            break
        time.sleep(poll)

    if not b.output_file_id:
        raise RuntimeError("Batch completed without output_file_id!")

    data: bytes = client.files.content(b.output_file_id).read()
    dest.write_bytes(data)
    print(f"[DONE] output → {dest}")


def parse_output(path: Path) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        cid = str(rec.get("custom_id", ""))

        if rec.get("error"):
            results[cid] = ""
            continue

        try:
            content = rec["response"]["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = ""

        results[cid] = content
    return results


###############################################################################
# Retrieval helper
###############################################################################

def retrieve_docs(queries: List[str], ds: Dataset, use_ref_date: bool) -> List[List[str]]:
    from reward import cached_retrieve_documents  # heavy import

    if use_ref_date:
        if "first_ref_date" in ds.column_names and "last_ref_date" in ds.column_names:
            mindates = ds["first_ref_date"]
            maxdates = ds["last_ref_date"]
        else:
            mindates = ["1945/01/01"] * len(queries)
            maxdates = ["2025/01/01"] * len(queries)
    else:
        if "min-date" in ds.column_names and "max-date" in ds.column_names:
            mindates = ds["min-date"]
            maxdates = ds["max-date"]
        else:
            mindates = ["1945/01/01"] * len(queries)
            maxdates = ["2025/01/01"] * len(queries)

    retrieved: List[List[str]] = [[] for _ in range(len(queries))]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {
            pool.submit(cached_retrieve_documents, q, mi, ma): idx
            for idx, (q, mi, ma) in enumerate(zip(queries, mindates, maxdates))
        }

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Retrieving documents"):
            i = futs[fut]
            try:
                retrieved[i] = fut.result()
            except Exception as e:
                print(f"[WARN] retrieval failed for idx {i}: {e}")
                retrieved[i] = []

    return retrieved


###############################################################################
# Append-safe writers
###############################################################################

def _mode(p: Path) -> str:
    return "a" if p.exists() else "w"


def append_jsonl(rows: Dataset, path: Path):
    with path.open(_mode(path), encoding="utf-8") as fp:
        for rec in rows:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_trec(tids: List[str], docs: List[List[str]], path: Path):
    with path.open(_mode(path), encoding="utf-8") as fp:
        for tid, dlist in zip(tids, docs):
            for rank, doc_id in enumerate(dlist[:1000]):
                fp.write(f"{tid} Q0 {doc_id} {rank + 1} {1000 - rank} run\n")


def f_beta(r: float, p: float, beta: float) -> float:
    if (beta * beta * p + r) == 0:
        return 0.0
    return (1 + beta * beta) * p * r / (beta * beta * p + r)


def append_eval(ds: Dataset, path: Path):
    with path.open(_mode(path), encoding="utf-8") as fp:
        for tid, rec, prec in zip(ds["topicid"], ds["recall"], ds["precision"]):
            f1 = f_beta(rec, prec, 1)
            f3 = f_beta(rec, prec, 3)
            f5 = f_beta(rec, prec, 5)
            fp.write(f"{tid}\t{rec:.4f}\t{prec:.4f}\t{f1:.4f}\t{f3:.4f}\t{f5:.4f}\n")





def read_eval_results(eval_path: Path) -> Tuple[List[float], List[float]]:
    """Read all recall and precision values from the eval file."""
    if not eval_path.exists():
        return [], [], [], [], []

    recalls = []
    precisions = []
    f1 = []
    f3 = []
    f5 = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("AVG"):
                continue

            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    _, recall, precision, f1_score, f3_score, f5_score = parts[:6]
                    recalls.append(float(recall))
                    precisions.append(float(precision))
                    f1.append(float(f1_score))
                    f3.append(float(f3_score))
                    f5.append(float(f5_score))

                except ValueError:
                    continue

    return recalls, precisions, f1, f3, f5


def write_eval_summary_from_file(eval_path: Path):
    """Write summary by reading all existing eval results."""
    recalls, precisions, f1, f3, f5 = read_eval_results(eval_path)

    if not recalls or not precisions:
        return

    # Calculate averages
    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    avg_f1 = sum(f1) / len(f1)
    avg_f3 = sum(f3) / len(f3)
    avg_f5 = sum(f5) / len(f5)
    # Write the summary line
    with eval_path.open("a", encoding="utf-8") as f:
        f.write(f"AVG\t{avg_recall:.4f}\t{avg_precision:.4f}\t{avg_f1:.4f}\t{avg_f3:.4f}\t{avg_f5:.4f}\n")



def append_round_success(round_num: int, successes: int, path: Path):
    header = "round_num\tnum_generations_to_success\n"

    # Read existing entries to avoid duplicates
    existing_rounds = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in lines[1:]:  # Skip header
                if line.strip():
                    existing_round = int(line.split("\t")[0])
                    existing_rounds.add(existing_round)

    # Only append if this round hasn't been recorded yet
    if round_num not in existing_rounds:
        mode = _mode(path)
        with path.open(mode, encoding="utf-8") as fp:
            if mode == "w":
                fp.write(header)
            fp.write(f"{round_num}\t{successes}\n")


###############################################################################
# Ledger helpers
###############################################################################

def read_ledger(path: Path) -> List[Tuple[int, str, str]]:
    if not path.exists():
        return []
    rows: List[Tuple[int, str, str]] = []
    for ln in path.read_text().splitlines():
        if ln.strip():
            r, b, s = ln.split("\t")
            rows.append((int(r), b, s))
    return rows


def append_ledger(path: Path, rnd: int, bid: str, status: str):
    with path.open("a", encoding="utf-8") as fp:
        fp.write(f"{rnd}\t{bid}\t{status}\n")


def mark_done(path: Path, bid: str):
    rows = read_ledger(path)
    with path.open("w", encoding="utf-8") as fp:
        for r, b, s in rows:
            if b == bid:
                s = "done"
            fp.write(f"{r}\t{b}\t{s}\n")


###############################################################################
# Round processing
###############################################################################

def process_round(
        out_file: Path,
        pending_rows: List[Dict],  # Only the rows that were processed in this round
        attempts: DefaultDict[str, int],
        solved: DefaultDict[str, bool],
        out_dir: Path,
        use_ref_date: bool,
) -> Tuple[List[float], List[float], int]:
    """
    Process a single round of batch results.
    Returns recall/precision lists for successful queries and count of newly solved topics.
    """
    out_map = parse_output(out_file)

    # Only process the pending rows that were actually in this batch
    successful_records = []

    for row in pending_rows:
        tid = str(row["topicid"])
        attempts[tid] += 1

        # Get the output for this topic
        output = out_map.get(tid, "")
        if not output:
            continue

        # Extract boolean query
        query = extract_boolean_query(output)
        if not query:
            continue

        # Add to successful records for retrieval
        successful_records.append({
            "row": row,
            "output": output,
            "query": query,
            "tid": tid
        })

    if not successful_records:
        return [], [], 0

    # Create dataset for retrieval (only successful queries)
    retrieval_ds = Dataset.from_list([rec["row"] for rec in successful_records])
    queries = [rec["query"] for rec in successful_records]

    # Retrieve documents
    retrieved = retrieve_docs(queries, retrieval_ds, use_ref_date)

    # Process results and save only successful ones
    final_successful_records = []
    recall_list = []
    precision_list = []
    newly_solved_count = 0

    for i, (rec, ret_docs) in enumerate(zip(successful_records, retrieved)):
        if not ret_docs:  # Skip if retrieval failed
            continue

        # Mark as solved and count
        tid = rec["tid"]
        if not solved[tid]:  # Only count if not previously solved
            solved[tid] = True
            newly_solved_count += 1

        # Compute metrics
        ground_truth = rec["row"]["ground_truth"]
        recall, precision = compute_recall_precision(ret_docs, ground_truth)

        # Store results
        final_successful_records.append({
            "topicid": rec["row"]["topicid"],
            "output": rec["output"],
            "boolean_query": rec["query"],
            "retrieved_ids": ret_docs,
            "recall": recall,
            "precision": precision,
        })

        recall_list.append(recall)
        precision_list.append(precision)

    # Save results (only successful ones)
    if final_successful_records:
        success_ds = Dataset.from_list(final_successful_records)
        append_jsonl(success_ds, out_dir / "generated_responses.jsonl")
        append_trec(success_ds["topicid"], success_ds["retrieved_ids"], out_dir / "result.trec")
        append_eval(success_ds, out_dir / "result.eval")

    return recall_list, precision_list, newly_solved_count


def load_solved_topic_ids(eval_path: Path) -> set[str]:
    """Reads the result.eval file and returns a set of topic IDs already evaluated."""
    if not eval_path.exists():
        return set()
    solved_ids = set()
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("AVG") or not line.strip():
                continue
            tid = line.split("\t")[0]
            solved_ids.add(str(tid))
    return solved_ids


###############################################################################
# MAIN
###############################################################################

def main(args):
    set_seed(42)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = out_dir / "bids.tsv"

    ds = load_dataset(args.dataset_name, split=args.split)
    id2row = {str(r["topicid"]): r for r in ds}

    attempts: DefaultDict[str, int] = defaultdict(int)
    solved: DefaultDict[str, bool] = defaultdict(bool)

    hist = read_ledger(ledger)
    all_recalls: List[float] = []
    all_precisions: List[float] = []

    # Load previously solved topics from existing eval file
    eval_path = out_dir / "result.eval"
    previously_solved_ids = load_solved_topic_ids(eval_path)
    for tid in previously_solved_ids:
        solved[tid] = True

    current_round = 0

    if hist:
        # Process pending batches ONE BY ONE in order
        # This ensures that the solved state is updated after each batch
        while True:
            # Find the next pending batch
            next_pending = None
            for round_num, batch_id, status in hist:
                current_round = max(current_round, round_num)
                if status == "pending":
                    next_pending = (round_num, batch_id)
                    break

            if not next_pending:
                break  # No more pending batches

            round_num, batch_id = next_pending
            out_file = out_dir / f"batch_output_round{round_num}.jsonl"
            print(f"[PENDING] Processing pending batch for round {round_num}")

            # Download the pending batch
            wait_and_download(batch_id, out_file, args.poll_seconds)
            mark_done(ledger, batch_id)

            # Determine which topics were supposed to be processed in this round
            # by using the current solved state (updated after each processed batch)
            round_pending_rows = [r for r in ds if not solved[str(r["topicid"])]]

            if round_pending_rows:
                print(f"[PROCESS] Processing round {round_num} with {len(round_pending_rows)} topics")

                recs, precs, newly_solved = process_round(
                    out_file, round_pending_rows, attempts, solved, out_dir, args.ref_date
                )
                all_recalls.extend(recs)
                all_precisions.extend(precs)

                append_round_success(round_num, newly_solved, out_dir / "generation_success_log.tsv")
                print(f"[ROUND {round_num}] Processed {newly_solved} newly solved topics")

            # Update hist status for this entry and reload ledger
            hist = [(r, b, "done" if b == batch_id else s) for r, b, s in hist]
            # Reload ledger to get updated status
            hist = read_ledger(ledger)

        # Then, process any existing done batches that might not have been processed yet
        for round_num, batch_id, status in hist:
            if status == "done":
                out_file = out_dir / f"batch_output_round{round_num}.jsonl"

                if not out_file.exists():
                    print(f"[WARN] Missing output file for completed round {round_num}, skipping")
                    continue

                # Check if this round's results are already in the eval file
                # by seeing if we can find any topic IDs from this round in the eval
                round_out_map = parse_output(out_file)
                round_topic_ids = set(round_out_map.keys())

                if round_topic_ids.issubset(previously_solved_ids):
                    print(f"[SKIP] Round {round_num} already processed")
                    continue

                # Determine which topics were supposed to be processed in this round
                temp_solved = set(previously_solved_ids)

                # Build up solved state from previous rounds
                for prev_round in range(1, round_num):
                    prev_out_file = out_dir / f"batch_output_round{prev_round}.jsonl"
                    if prev_out_file.exists():
                        prev_out_map = parse_output(prev_out_file)
                        for r in ds:
                            tid = str(r["topicid"])
                            if tid in temp_solved:
                                continue
                            output = prev_out_map.get(tid, "")
                            if output and extract_boolean_query(output):
                                temp_solved.add(tid)

                round_pending_rows = [r for r in ds if str(r["topicid"]) not in temp_solved]

                if round_pending_rows:
                    print(f"[REPROCESS] Processing done round {round_num} with {len(round_pending_rows)} topics")

                    recs, precs, newly_solved = process_round(
                        out_file, round_pending_rows, attempts, solved, out_dir, args.ref_date
                    )
                    all_recalls.extend(recs)
                    all_precisions.extend(precs)

                    append_round_success(round_num, newly_solved, out_dir / "generation_success_log.tsv")
                    print(f"[ROUND {round_num}] Reprocessed {newly_solved} newly solved topics")

    # If no history, start from round 0
    if not hist:
        current_round = 0
    max_total_rounds = args.max_rounds
    if current_round >= max_total_rounds:
        print(f"[INFO] Already completed {current_round} rounds, max_rounds={max_total_rounds}. Skipping new rounds.")
    else:
    # Process additional rounds
        for rnd in range(current_round + 1, min(current_round + 1 + args.max_rounds, max_total_rounds + 1)):
            pending_rows = [id2row[tid] for tid in id2row if not solved[tid]]
            if not pending_rows:
                print("[✓] All topics solved — exiting.")
                break

            print(f"\n### ROUND {rnd} – {len(pending_rows)} pending topics")

            inp_path = out_dir / f"batch_input_round{rnd}.jsonl"
            make_batch_input(
                pending_rows,
                args.openai_model,
                args.temperature,
                args.max_prompt_length,
                args.max_completion_length,
                inp_path,
            )

            bid = submit_batch(inp_path)
            append_ledger(ledger, rnd, bid, "pending")

            out_file = out_dir / f"batch_output_round{rnd}.jsonl"
            wait_and_download(bid, out_file, args.poll_seconds)

            recs, precs, newly_solved = process_round(
                out_file, pending_rows, attempts, solved, out_dir, args.ref_date
            )
            all_recalls.extend(recs)
            all_precisions.extend(precs)

            mark_done(ledger, bid)
            append_round_success(rnd, newly_solved, out_dir / "generation_success_log.tsv")

    # Write final summary
    write_eval_summary_from_file(out_dir / "result.eval")

    print(
        f"[SUMMARY] {sum(solved.values())}/{len(ds)} topics solved • "
        f"total rounds in ledger: {len(read_ledger(ledger))}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Resumable GPT-4o Batch evaluation")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--openai_model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--output_dir", default="runs/iter_batch")
    parser.add_argument("--poll_seconds", type=int, default=30)
    parser.add_argument("--max_rounds", type=int, default=10)
    parser.add_argument("--ref_date", action="store_true", help="Use reference dates for retrieval")
    main(parser.parse_args())