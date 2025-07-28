import xml.etree.ElementTree as ET
import glob
import os
import json
from tqdm import tqdm
from datetime import datetime

def load_reviews(input_folder):
    review_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    reviews = {}
    for file_path in tqdm(review_files, desc="Loading reviews"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    pmid = str(review.get("pmid"))
                    if pmid:
                        reviews[pmid] = review
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {file_path}: {line.strip()}")
    return reviews


def load_ids(input_id_file):
    if not os.path.exists(input_id_file):
        print(f"Warning: {input_id_file} does not exist. Proceeding without filtering by IDs.")
        return None

    with open(input_id_file, 'r') as f:
        id_set = set(str(line.strip()) for line in f if line.strip())
    print(f"Loaded {len(id_set)} IDs from {input_id_file}.")
    return id_set


def process_reviews(loaded_reviews, input_id_file, output_folder):
    id_set = load_ids(input_id_file)
    os.makedirs(output_folder, exist_ok=True)
    filename_out = "all.jsonl"
    output_path = os.path.join(output_folder, filename_out)
    if os.path.exists(output_path):
        print(f"Skipping {filename_out}, already processed.")
        return
    with open(output_path, "w", encoding="utf-8") as f_out:
        for pmid in tqdm(id_set, desc="Processing reviews"):
            if pmid not in loaded_reviews:
                #print(f"PMID {pmid} not found in loaded reviews.")
                continue
            review = loaded_reviews.get(pmid)
            references_pmids = review.get("references-pmids", [])
            if len(references_pmids) == 0:
                continue
            review_entry = {
                "pmid": pmid,
                "pmc-id": review.get("pmc-id", ""),
                "title": review.get("title", ""),
                "max-date": review.get("date", "") if review.get("date") else review.get("max-date", ""),
                "references-pmids": review.get("references-pmids", [])
            }
            json.dump(review_entry, f_out)
            f_out.write("\n")



def main():
    print("Starting to process PubMed reviews...")
    input_folder_reviews = "../data/processed/pubmed/all_review_types"

    reviews = load_reviews(input_folder_reviews)

    input_id_sr ="../data/pubmed_ids.txt"
    output_folder_sr = "../data/processed/pubmed/sr"

    print("Processing systematic reviews...")
    process_reviews(reviews, input_id_sr, output_folder_sr)




if __name__ == "__main__":
    main()