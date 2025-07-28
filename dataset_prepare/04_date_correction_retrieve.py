from tqdm import tqdm
import json
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from utils.pubmed_submission import retrieve_documents
def load_all_docs(input_folder, max_workers=12):
    """Load all articles (PMID → date)"""
    files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    data = {}

    def load_file(file_path):
        d = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    pmid = str(item.get("pmid"))
                    if pmid:
                        d[pmid] = item.get("date", "")
                except json.JSONDecodeError:
                    continue
        return d

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(load_file, files), total=len(files), desc="Loading all_collection"))
    for r in results:
        data.update(r)
    return data
from datetime import datetime, timedelta
def get_first_and_last_valid_dates(ref_pmids, all_dates):
    """Return the first and last valid date from the reference PMIDs, shifted by +1 day and -1 day respectively."""
    dates = [all_dates.get(pmid) for pmid in ref_pmids]
    dates = [d for d in dates if d]  # Remove None/empty
    if not dates:
        return None, None

    try:
        parsed_dates = [datetime.strptime(d, "%Y/%m/%d") for d in dates]
        first_date = min(parsed_dates) + timedelta(days=1)
        last_date = max(parsed_dates) + timedelta(days=1)
        return first_date.strftime("%Y/%m/%d"), last_date.strftime("%Y/%m/%d")
    except ValueError:
        print(f"Error parsing dates for PMIDs: {ref_pmids}")
        return None, None
def process_and_rewrite_reviews(sr_folder, all_doc_dates, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    input_files = glob.glob(os.path.join(sr_folder, "*.jsonl"))

    for file_path in tqdm(input_files, desc="Processing SR reviews"):
        filename = os.path.basename(file_path)
        out_path = os.path.join(output_folder, filename)

        with open(file_path, 'r', encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
            for line in tqdm(fin):
                try:
                    review = json.loads(line)
                    #review_date = review.get("max-date", "") if review.get("max-date") else review.get("date", "")
                    ref_pmids = review.get("references-pmids", [])
                    first_ref_date, last_ref_date = get_first_and_last_valid_dates(ref_pmids, all_doc_dates)
                    review["first_ref_date"] = first_ref_date
                    review["last_ref_date"] = last_ref_date
                    boolean_query = review.get("boolean_query", "")
                    if boolean_query:
                        retrieved_ids = retrieve_documents(boolean_query, (first_ref_date, last_ref_date))
                        print(f"Retrieved {len(retrieved_ids)} documents for query")
                        review["retrieved_ids"] = retrieved_ids

                    fout.write(json.dumps(review) + "\n")
                except json.JSONDecodeError:
                    continue

if __name__ == "__main__":
    #input_folder_sr = "../data/processed/pubmed/sr"
    #input_folder_sr = "../data/processed/clef"
    input_folder_sr = "../data/processed/seed"
    input_folder_all_collection = "../data/processed/pubmed/all_collection"
    #output_folder = "../data/processed/pubmed/sr"
    #output_folder = "../data/processed/clef_augmented"
    output_folder = "../data/processed/seed_augmented"
    all_doc_dates = load_all_docs(input_folder_all_collection)
    process_and_rewrite_reviews(input_folder_sr, all_doc_dates, output_folder)

    print(f"Augmented SR files with `first_ref_date` and `last_ref_date` written to: {output_folder}")
