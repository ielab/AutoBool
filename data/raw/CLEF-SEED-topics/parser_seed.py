import json
import time
import re
import unicodedata
from Bio import Entrez
import os
import socket
socket.setdefaulttimeout(10)  # ⏱️ Timeout set here (10 seconds max)
Entrez.email = "your.email@example.com"
OUTPUT_FILE = "seed_collection_with_pmid.jsonl"
INPUT_FILE = "seed_collection.jsonl"

def clean_title(title):
    normalized = unicodedata.normalize("NFKD", title)
    ascii_cleaned = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^\w\s\-.,:;()/&]+", "", ascii_cleaned)

def build_pubmed_query(title):
    cleaned_title = clean_title(title)
    return f"{cleaned_title}[ti] AND 2000/01/01:2022/01/01[PDAT]"

def get_top_two_pubmed_results(original_title):
    try:
        query = build_pubmed_query(original_title)
        handle = Entrez.esearch(db="pubmed", term=query, retmax=2)
        result = Entrez.read(handle)
        handle.close()

        pmids = result.get("IdList", [])
        if not pmids:
            return []

        summary_handle = Entrez.esummary(db="pubmed", id=",".join(pmids))
        summary = Entrez.read(summary_handle)
        summary_handle.close()

        return [(entry["Id"], entry["Title"]) for entry in summary]

    except Exception as e:
        print(f"Error querying title '{original_title}': {e}")
        return []

def load_completed_pmids(output_file):
    completed = {}
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if "title" in item:
                    completed[item["title"]] = item
    return completed

def write_item(output_file, item):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(item) + "\n")

def process_all(input_file, output_file):
    completed = load_completed_pmids(output_file)

    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            review_title = item.get("title")

            if review_title in completed:
                continue  # Skip already processed

            print("\n======================")
            print(f"Review Title:\n> {review_title}")

            top_matches = get_top_two_pubmed_results(review_title)

            if not top_matches:
                print("No results found.")
                item["pmids"] = []
                write_item(output_file, item)
                continue

            for i, (pmid, title) in enumerate(top_matches, 1):
                print(f"{i}. PMID: {pmid}")
                print(f"   Title: {title}")

            decision = input("Select 1 / 2 / y (both) / n (none): ").strip().lower()
            if decision == '1':
                item["pmids"] = [top_matches[0][0]]
            elif decision == '2':
                item["pmids"] = [top_matches[1][0]]
            elif decision == 'y':
                item["pmids"] = [pmid for pmid, _ in top_matches]
            else:
                item["pmids"] = []

            write_item(output_file, item)

def manual_fix_unmatched(output_file):
    # Load all
    items = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    fixed_items = []
    for item in items:
        if not item.get("pmids"):
            print("\n======================")
            print(f"Manual entry for title:\n> {item.get('title')}")
            user_input = input("Enter PMIDs as [123,456] or n to leave empty: ").strip()

            if user_input.lower() == 'n':
                item["pmids"] = []
            else:
                try:
                    pmids = json.loads(user_input)
                    if isinstance(pmids, list):
                        item["pmids"] = pmids
                    else:
                        print("Invalid format. Keeping empty.")
                        item["pmids"] = []
                except Exception:
                    print("Could not parse input. Keeping empty.")
                    item["pmids"] = []

        fixed_items.append(item)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in fixed_items:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    process_all(INPUT_FILE, OUTPUT_FILE)
    manual_fix_unmatched(OUTPUT_FILE)
