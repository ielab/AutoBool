import os
import json
from tqdm import tqdm

if __name__ == "__main__":
    input_jsonl_file = "../data/raw/CLEF-SEED-topics/CLEF-2018.jsonl"
    input_date_file = "../data/raw/CLEF-SEED-topics/combined_pubdates"
    input_qrel_file = "../data/raw/CLEF-SEED-topics/qrels_2018.txt"

    output_folder = "../data/processed/clef"
    output_file = os.path.join(output_folder, "all.jsonl")

    os.makedirs(output_folder, exist_ok=True)

    # Load date file
    print("Loading date file...")
    date_dict = {}
    with open(input_date_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading pubdates"):
            pmid, min_date, max_date = line.strip().split()
            # adding / to the date format
            min_date = min_date[0:4] + "/" + min_date[4:6] + "/" + min_date[6:8]
            max_date = max_date[0:4] + "/" + max_date[4:6] + "/" + max_date[6:8]
            date_dict[pmid] = {
                "min-date": min_date,
                "max-date": max_date
            }
    print(f"Loaded dates for {len(date_dict)} PMIDs")

    # Load qrel file
    print("Loading qrels...")
    qrel_dict = {}
    with open(input_qrel_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading qrels"):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            pmid = parts[0]
            relevance = int(parts[3])
            doc_id = parts[2]

            if relevance > 0:
                if pmid not in qrel_dict:
                    qrel_dict[pmid] = set()
                qrel_dict[pmid].add(doc_id)
    print(f"Loaded qrels for {len(qrel_dict)} topics")

    # Load and process reviews
    print("Processing reviews...")
    reviews = []
    with open(input_jsonl_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading JSONL reviews"):
            try:
                review = json.loads(line)
                review_id = str(review.get("topicid", ""))
                dates = date_dict.get(review_id, {})

                new_dict = {
                    "topicid" : review.get("topicid", ""),
                    "title": review.get("title", ""),
                    "min-date": dates.get("min-date", ""),
                    "max-date": dates.get("max-date", ""),
                    "boolean_query": review.get("original_query", ""),
                    "references-pmids": list(qrel_dict.get(review_id, []))
                }
                if len(new_dict["references-pmids"]) == 0:
                    continue
                reviews.append(new_dict)
            except json.JSONDecodeError:
                continue
    print(f"Processed {len(reviews)} valid reviews with references")

    # Write output
    print(f"Writing output to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for review in tqdm(reviews, desc="Writing JSONL"):
            json.dump(review, f_out)
            f_out.write("\n")

    print("Done.")
