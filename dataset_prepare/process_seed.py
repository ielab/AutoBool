import os
import json
from tqdm import tqdm


def correct_date(date):
    date_split = date.split("/")
    date_corrected = date_split[2] + "/" + date_split[1] + "/" + date_split[0]
    return date_corrected


if __name__ == "__main__":
    input_jsonl_file = "../data/raw/CLEF-SEED-topics/seed_collection.jsonl"
    input_qrel_file = "../data/raw/CLEF-SEED-topics/seed_collection.qrels"

    output_folder = "../data/processed/seed"
    output_file = os.path.join(output_folder, "all.jsonl")

    os.makedirs(output_folder, exist_ok=True)


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
                min_date = correct_date(review.get("Date From", ""))
                max_date = correct_date(review.get("Date Run", ""))

                new_dict = {
                    "topicid" : review.get("topicid", ""),
                    "title": review.get("title", ""),
                    "min-date": min_date,
                    "max-date": max_date,
                    "boolean_query": review.get("Edited search", ""),
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
