import xml.etree.ElementTree as ET
import glob
import os
import json
from tqdm import tqdm
from datetime import datetime


def check_id(article, id_set=None):
    pmid_elem = article.find('.//PMID')
    if pmid_elem is None:
        return False
    pmid = pmid_elem.text.strip()
    return id_set is None or pmid in id_set


def extract_metadata_review(article):
    pmid_elem = article.find('.//PMID')
    if pmid_elem is None:
        return None
    pmid = pmid_elem.text.strip()

    # PMC ID
    pmc_id = None
    for id_elem in article.findall('.//ArticleId'):
        if id_elem.attrib.get("IdType") == "pmc":
            pmc_id = id_elem.text.strip()
            break
    if pmc_id is None:
        return None

    # Title
    title_elem = article.find('.//ArticleTitle')
    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

    # Abstract
    abstract_elems = article.findall('.//Abstract/AbstractText')
    abstract_text = " ".join(elem.text.strip() for elem in abstract_elems if elem.text)

    # Publication date
    date_elem = article.find('.//ArticleDate')
    if date_elem is not None:
        year = date_elem.findtext('Year')
        month = date_elem.findtext('Month') or "01"
        day = date_elem.findtext('Day') or "01"
    else:
        # Fallback to Journal > PubDate
        date_elem = article.find('.//PubDate')
        year = date_elem.findtext('Year')
        month = date_elem.findtext('Month') or "01"
        day = date_elem.findtext('Day') or "01"

    try:
        pub_date = datetime.strptime(f"{year}/{month}/{day}", "%Y/%b/%d").strftime("%Y/%m/%d")
    except ValueError:
        try:
            pub_date = datetime.strptime(f"{year}/{month}/{day}", "%Y/%m/%d").strftime("%Y/%m/%d")
        except ValueError:
            pub_date = None

    # References
    pmids = []
    pmcids = []
    others = []
    for ref in article.findall('.//ReferenceList/Reference'):
        found = False
        for ref_id in ref.findall('.//ArticleId'):
            id_type = ref_id.attrib.get("IdType")
            id_text = ref_id.text.strip() if ref_id.text else None
            if id_text:
                if id_type == "pubmed":
                    pmids.append(id_text)
                    found = True
                elif id_type == "pmc":
                    pmcids.append(id_text)
                    found = True
                else:
                    others.append(id_text)
        if not found:
            cit = ref.findtext('Citation')
            if cit:
                others.append(cit.strip())

    if not (pmids or pmcids):
        return None

    return {
        "pmid": pmid,
        "pmc-id": pmc_id,
        "title": title,
        "abstract": abstract_text,
        "max-date": pub_date,
        "references-pmids": pmids,
        "references-pmcids": pmcids,
        "references_others": others
    }


def extract_matadata_others(article):
    # this extract meta data of other articles that are not in id_set, only require pmid, title, abstract and date
    pmid_elem = article.find('.//PMID')
    if pmid_elem is None:
        return None
    pmid = pmid_elem.text.strip()
    for id_elem in article.findall('.//ArticleId'):
        if id_elem.attrib.get("IdType") == "pmc":
            pmc_id = id_elem.text.strip()
            break


    title_elem = article.find('.//ArticleTitle')
    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""
    abstract_elems = article.findall('.//Abstract/AbstractText')
    abstract_text = " ".join(elem.text.strip() for elem in abstract_elems if elem.text)
    date_elem = article.find('.//ArticleDate')
    if date_elem is not None:
        year = date_elem.findtext('Year')
        month = date_elem.findtext('Month') or "01"
        day = date_elem.findtext('Day') or "01"
    else:
        # Fallback to Journal > PubDate
        date_elem = article.find('.//PubDate')
        year = date_elem.findtext('Year')
        month = date_elem.findtext('Month') or "01"
        day = date_elem.findtext('Day') or "01"

    try:
        pub_date = datetime.strptime(f"{year}/{month}/{day}", "%Y/%b/%d").strftime("%Y/%m/%d")
    except ValueError:
        try:
            pub_date = datetime.strptime(f"{year}/{month}/{day}", "%Y/%m/%d").strftime("%Y/%m/%d")
        except ValueError:
            pub_date = None
    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract_text,
        "date": pub_date
    }


def parse_articles(xml_path, id_set=None):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    reviews = []
    all_articles = []

    for article in root.findall('.//PubmedArticle'):
        if check_id(article, id_set=id_set):
            review = extract_metadata_review(article)
            if review:
                reviews.append(review)
                all_articles.append({
                    "pmid": review["pmid"],
                    "title": review["title"],
                    "abstract": review["abstract"],
                    "date": review["date"],
                })
                if id_set is not None:
                    id_set.discard(review["pmid"])
        else:
            other_article = extract_matadata_others(article)
            if other_article:
                all_articles.append(other_article)

    return reviews, all_articles


def main():
    input_folder = "../data/raw/pubmed/"
    output_folder_reviews = "../data/processed/pubmed/sr_augmented_result"
    output_folder_all = "../data/processed/pubmed/all_collection"
    input_ids_path = "../data/pubmed_ids.txt"

    os.makedirs(output_folder_reviews, exist_ok=True)
    os.makedirs(output_folder_all, exist_ok=True)

    # Load PMIDs
    if os.path.exists(input_ids_path):
        with open(input_ids_path, 'r') as f:
            id_set = set(line.strip() for line in f if line.strip())
    else:
        print(f"Warning: {input_ids_path} does not exist. Proceeding without filtering by IDs.")
        id_set = None

    input_files = glob.glob(f"{input_folder}/*.xml")
    real_input_files = []
    for file_path in tqdm(input_files):
        filename_out = os.path.basename(file_path).replace(".xml", ".jsonl")
        review_out_path = os.path.join(output_folder_reviews, filename_out)
        all_out_path = os.path.join(output_folder_all, filename_out)
        if not (os.path.exists(all_out_path)):
            real_input_files.append(file_path)

    print("Parsing XML files...")
    for file_path in tqdm(real_input_files, desc="Processing files"):
        filename_out = os.path.basename(file_path).replace(".xml", ".jsonl")
        review_out_path = os.path.join(output_folder_reviews, filename_out)
        all_out_path = os.path.join(output_folder_all, filename_out)
        # if the file already exists, skip it
        if os.path.exists(review_out_path) and os.path.exists(all_out_path):
            print(f"Skipping {filename_out}, already processed.")
            continue
        matched_review_articles, all_articles = parse_articles(file_path, id_set=id_set)
        print(f"Processed {filename_out}: {len(matched_review_articles)} review articles, {len(all_articles)} total articles.")
        if matched_review_articles:
            with open(review_out_path, "w", encoding="utf-8") as f_out:
                for entry in matched_review_articles:
                    json.dump(entry, f_out)
                    f_out.write("\n")
        if all_articles:
            with open(all_out_path, "w", encoding="utf-8") as f_out:
                for entry in all_articles:
                    json.dump(entry, f_out)
                    f_out.write("\n")

    print("Done.")


if __name__ == "__main__":
    main()