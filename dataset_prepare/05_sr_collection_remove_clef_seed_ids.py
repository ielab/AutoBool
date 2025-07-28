import json
import os
from tqdm import tqdm

input_sr_augmented = "../data/processed/pubmed/sr_augmented/all.jsonl"
output_sr_collection = "../data/processed/pubmed/sr_augmented/all_filtered.jsonl"
input_remove_files = [
    "/scratch3/wan458/evidence_synthesis_benchmark/data/raw/CLEF-SEED-topics/CLEF-2018_with_pmids.jsonl",
    "/scratch3/wan458/evidence_synthesis_benchmark/data/raw/CLEF-SEED-topics/seed_collection_with_pmid.jsonl"
]

# first read the two input remove files and collect pmids to remove
def collect_pmids_to_remove(files):
    pmids_to_remove = set()
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                pmids = item.get("pmids", [])
                pmids = [str(pmid) for pmid in pmids]
                pmids_to_remove.update(pmids)
    return pmids_to_remove

def filter_sr_collection(input_file, pmids_to_remove, output_file):
    counter = 0
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            item = json.loads(line)
            pmid = str(item.get("pmid"))
            if pmid in pmids_to_remove:
                counter += 1
                continue
            fout.write(json.dumps(item) + "\n")
    print(f"Removed {counter} items from the SR collection based on PMIDs.")

if __name__ == "__main__":
    pmids_to_remove = collect_pmids_to_remove(input_remove_files)
    print(f"Collected {len(pmids_to_remove)} PMIDs to remove.")

    filter_sr_collection(input_sr_augmented, pmids_to_remove, output_sr_collection)
    print(f"Filtered SR collection saved to {output_sr_collection}.")
