"""
steps.py - Individual step functions for PubMed processing pipeline
"""
import os
import json
import requests
import gzip
import shutil
import xml.etree.ElementTree as ET
import glob
from tqdm import tqdm
from datetime import datetime, timedelta
from multiprocessing import Pool, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Union
import time
import sys

from utils.pubmed_submission import pubmed_submission, retrieve_documents
from config import *


def step_00_get_pubmed_ids(query=DEFAULT_QUERY, output_file=PUBMED_IDS_FILE):
    """
    Step 0: Query PubMed API to get systematic review PMIDs
    """
    print(f"🔍 Step 0: Getting PubMed IDs for query: {query}")

    with open(output_file, "w") as f:
        ids, counter_too_many = pubmed_submission(query, None, 0)
        for id in ids:
            f.write(id + "\n")

    print(f"✅ Step 0 complete: {len(ids)} PMIDs saved to {output_file}")
    return output_file


def step_01_download_pubmed(
    base_url=PUBMED_BASE_URL,
    dest_dir=PUBMED_RAW_DIR,
    start_index=PUBMED_START_INDEX,
    max_index=PUBMED_MAX_INDEX,
    test_first=True
):
    """
    Step 1: Download and decompress PubMed baseline files
    """
    print(f"📥 Step 1: Downloading PubMed baseline files to {dest_dir}")

    os.makedirs(dest_dir, exist_ok=True)

    # Test connection first
    if test_first:
        print(f"🔍 Testing connection to {base_url}")
        try:
            test_response = requests.get(base_url, timeout=10)
            if test_response.status_code == 200:
                print(f"✅ Connection successful to PubMed FTP")
            else:
                print(f"⚠️  Warning: Got status {test_response.status_code} from {base_url}")
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            print("Continuing anyway...")

    def download_and_decompress(file_index):
        file_name = f"pubmed25n{file_index:04d}.xml.gz"
        xml_name = file_name[:-3]
        url = f"{base_url}{file_name}"
        gz_path = os.path.join(dest_dir, file_name)
        xml_path = os.path.join(dest_dir, xml_name)

        # Skip if already decompressed
        if os.path.exists(xml_path):
            print(f"✅ File {xml_name} already exists, skipping")
            return True

        print(f"🔍 Attempting to download: {url}")

        # Try to download the file
        try:
            response = requests.get(url, stream=True, timeout=200)
            print(f"📡 Response status: {response.status_code}")

            if response.status_code == 200:
                file_size = response.headers.get('content-length')
                if file_size:
                    file_size = int(file_size)
                    print(f"📦 Downloading {file_name} ({file_size/1024/1024:.1f} MB)")

                with open(gz_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"🗜️  Decompressing {file_name}")
                # Decompress .gz to .xml
                with gzip.open(gz_path, 'rb') as f_in, open(xml_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

                os.remove(gz_path)  # Optional: remove .gz file after decompression
                print(f"✅ Successfully processed {xml_name}")
                return True
            elif response.status_code == 404:
                print(f"❌ File not found (404): {url}")
                return False  # File not found (likely end of available shards)
            else:
                print(f"❌ HTTP Error {response.status_code}: {url}")
                return False
        except requests.exceptions.Timeout:
            print(f"⏱️  Timeout downloading {url}")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"🌐 Connection error downloading {url}: {e}")
            return False
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
            return False

    print("Starting download of PubMed baseline shards...")
    for i in tqdm(range(start_index, max_index + 1)):
        success = download_and_decompress(i)
        if not success:
            print(f"Stopping at index {i:04d}")
            break

    print(f"✅ Step 1 complete: PubMed files downloaded to {dest_dir}")
    return dest_dir


# Wrapper function for multiprocessing (needed to unpack arguments)
def _process_xml_wrapper(args):
    return _parse_xml_file_iter(*args)

# Helper functions for step 02 (moved outside to avoid pickle issues)
def _parse_article_basic(article):
    pmid = article.findtext('.//PMID', '').strip()
    if not pmid:
        return None

    title_elem = article.find('.//ArticleTitle')
    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

    abstract_elems = article.findall('.//Abstract/AbstractText')
    abstract = " ".join(e.text.strip() for e in abstract_elems if e.text)

    date_elem = article.find('.//ArticleDate') or article.find('.//PubDate')
    year = date_elem.findtext('Year', '')
    month = date_elem.findtext('Month', '01')
    day = date_elem.findtext('Day', '01')

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
        "abstract": abstract,
        "max-date": pub_date
    }

def _parse_article_review(article):
    meta = _parse_article_basic(article)
    if meta is None:
        return None

    for id_elem in article.findall('.//ArticleId'):
        if id_elem.attrib.get("IdType") == "pmc":
            meta["pmcid"] = id_elem.text.strip()
            break
    else:
        meta["pmcid"] = None

    return meta

def _parse_xml_file_iter(file_path, id_dict, output_folder_reviews, output_folder_all):
    fname = os.path.basename(file_path).replace(".xml", ".jsonl")
    review_out = os.path.join(output_folder_reviews, fname)
    all_out = os.path.join(output_folder_all, fname)

    if os.path.exists(review_out) and os.path.exists(all_out):
        return

    reviews = []
    all_articles = []

    context = ET.iterparse(file_path, events=("end",))
    for event, elem in context:
        if elem.tag == 'PubmedArticle':
            pmid = elem.findtext('.//PMID', '').strip()
            if not pmid:
                elem.clear()
                continue

            is_review = pmid in id_dict
            meta = _parse_article_review(elem) if is_review else _parse_article_basic(elem)

            if meta:
                if is_review:
                    reviews.append(meta)
                    id_dict.pop(pmid, None)
                all_articles.append(meta)

            elem.clear()

    if all_articles:
        with open(all_out, "w", encoding="utf-8") as f:
            for entry in all_articles:
                json.dump(entry, f)
                f.write("\n")

    if reviews:
        with open(review_out, "w", encoding="utf-8") as f:
            for entry in reviews:
                json.dump(entry, f)
                f.write("\n")

def step_02_pmcid_date_mapping(
    input_folder=PUBMED_RAW_DIR,
    output_base=PUBMED_PROCESSED_DIR,
    input_ids_path=PUBMED_IDS_FILE,
    num_threads=DEFAULT_NUM_THREADS
):
    """
    Step 2: Extract PMC IDs and dates from PubMed XML files
    """
    print(f"🗂️  Step 2: Creating PMC ID and date mapping from {input_folder}")

    output_folder_reviews = os.path.join(output_base, "sr_mapping")
    output_folder_all = os.path.join(output_base, "all_collection")

    os.makedirs(output_folder_reviews, exist_ok=True)
    os.makedirs(output_folder_all, exist_ok=True)

    if os.path.exists(input_ids_path):
        with open(input_ids_path, 'r') as f:
            id_list = [line.strip() for line in f if line.strip()]
    else:
        id_list = []

    manager = Manager()
    id_dict = manager.dict({pmid: True for pmid in id_list})

    input_files = [
        f for f in glob.glob(f"{input_folder}/*.xml")
        if not (
            os.path.exists(os.path.join(output_folder_reviews, os.path.basename(f).replace(".xml", ".jsonl"))) and
            os.path.exists(os.path.join(output_folder_all, os.path.basename(f).replace(".xml", ".jsonl")))
        )
    ]

    print(f"Processing {len(input_files)} files using {num_threads} threads...")
    print(f"Target review PMIDs: {len(id_dict)}")

    args_list = [
        (f, id_dict, output_folder_reviews, output_folder_all)
        for f in input_files
    ]

    with Pool(processes=num_threads) as pool:
        list(tqdm(pool.imap_unordered(_process_xml_wrapper, args_list), total=len(args_list)))

    # Aggregate only systematic review files into single file
    print("🔄 Aggregating systematic review .jsonl files...")

    sr_all_file = os.path.join(output_folder_reviews, "all.jsonl")
    sr_jsonl_files = glob.glob(os.path.join(output_folder_reviews, "*.jsonl"))
    sr_jsonl_files = [f for f in sr_jsonl_files if not f.endswith("all.jsonl")]  # Exclude existing all.jsonl

    if sr_jsonl_files:
        with open(sr_all_file, 'w', encoding='utf-8') as outfile:
            for jsonl_file in tqdm(sr_jsonl_files, desc="Aggregating SR files"):
                with open(jsonl_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        outfile.write(line)
        print(f"📁 Aggregated {len(sr_jsonl_files)} SR files into {sr_all_file}")
    else:
        print("📁 No SR files found to aggregate")

    print(f"✅ Step 2 complete: Processed {len(input_files)} files")
    if len(id_dict) > 0:
        print(f"⚠️  {len(id_dict)} PMIDs were not matched.")

    return output_folder_reviews, output_folder_all


def step_03_remove_clef_seed_ids(
    input_sr_file=None,
    output_file=None,
    remove_files=CLEF_SEED_FILES
):
    """
    Step 3: Remove CLEF seed IDs from systematic review collection
    """
    if input_sr_file is None:
        # Create combined file first
        sr_dir = SR_MAPPING_DIR
        input_sr_file = sr_dir / "all.jsonl"

        # Combine all JSONL files in sr_mapping directory
        with open(input_sr_file, 'w', encoding='utf-8') as outfile:
            for jsonl_file in glob.glob(str(sr_dir / "*.jsonl")):
                if jsonl_file != str(input_sr_file):
                    with open(jsonl_file, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())

    if output_file is None:
        output_file = SR_MAPPING_DIR / "removed_clef_seed.jsonl"

    print(f"🧹 Step 3: Removing CLEF seed IDs from {input_sr_file}")

    def collect_pmids_to_remove(files):
        pmids_to_remove = set()
        for file in files:
            if os.path.exists(file):
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            pmids = item.get("pmids", [])
                            pmids = [str(pmid) for pmid in pmids]
                            pmids_to_remove.update(pmids)
                        except json.JSONDecodeError:
                            continue
        return pmids_to_remove

    def filter_sr_collection(input_file, pmids_to_remove, output_file):
        counter = 0
        with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                try:
                    item = json.loads(line)
                    pmid = str(item.get("pmid"))
                    if pmid in pmids_to_remove:
                        counter += 1
                        continue
                    fout.write(json.dumps(item) + "\n")
                except json.JSONDecodeError:
                    continue
        print(f"Removed {counter} items from the SR collection based on PMIDs.")

    pmids_to_remove = collect_pmids_to_remove(remove_files)
    print(f"Collected {len(pmids_to_remove)} PMIDs to remove.")

    filter_sr_collection(input_sr_file, pmids_to_remove, output_file)
    print(f"✅ Step 3 complete: Filtered SR collection saved to {output_file}")

    return output_file


def step_04_get_references(
    input_file=None,
    output_file=None,
    api_key=DEFAULT_API_KEY,
    xml_dump_dir=XML_DUMP_DIR,
    batch_size=DEFAULT_BATCH_SIZE,
    max_workers=DEFAULT_MAX_WORKERS
):
    """
    Step 4: Extract reference PMIDs from systematic reviews
    """
    if input_file is None:
        input_file = SR_MAPPING_DIR / "removed_clef_seed.jsonl"
    if output_file is None:
        output_file = SR_MAPPING_DIR / "filtered_references.jsonl"

    print(f"📚 Step 4: Extracting references from {input_file}")

    # Helper functions (keeping the same logic from original script)
    def chunked(iterable, n):
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                break
            yield batch

    def _normalize_title(text: str) -> str:
        return (text or "").strip().lower()

    def fetch_articles_xml_batch(pmcids: List[str], api_key: Optional[str] = None) -> Dict[str, str]:
        if not pmcids:
            return {}
        params = {"db": "pmc", "id": ",".join(cid.lstrip("PMC") for cid in pmcids), "retmode": "xml"}
        if api_key:
            params["api_key"] = api_key

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(EFETCH_URL, params=params, headers=HEADERS, timeout=TIMEOUT)
                r.raise_for_status()
                if r.text.strip().startswith("Error"):
                    return {}
                return _split_articles_by_pmcid(r.text)
            except (requests.RequestException, requests.Timeout) as exc:
                if attempt == MAX_RETRIES:
                    print(f"[WARN] Batch fetch failed for {pmcids[:3]}… ({len(pmcids)} IDs): {exc}", file=sys.stderr)
                    return {}
                time.sleep(2 * attempt)
        return {}

    def _split_articles_by_pmcid(xml_text: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return out
        for art in root.findall(".//article"):
            pmc_el = art.find(".//article-id[@pub-id-type='pmcid']")
            if pmc_el is not None and pmc_el.text:
                out[pmc_el.text.strip()] = ET.tostring(art, encoding="unicode")
        return out

    def extract_reference_pmids_by_section(xml_text: str) -> Tuple[Set[str], Set[str]]:
        """Returns (results-section-pmids, all-reference-pmids)"""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return set(), set()

        all_refs: Dict[str, Set[str]] = {}  # {ref-id: set of pmids}
        for ref in root.findall(".//ref"):
            rid = ref.attrib.get("id", "")
            pmids = {pub.text.strip() for pub in ref.findall('.//pub-id[@pub-id-type="pmid"]') if pub.text}
            if rid and pmids:
                all_refs[rid] = pmids

        all_pmids = {pmid for pmids in all_refs.values() for pmid in pmids}

        results_pmids: Set[str] = set()
        for sec in root.findall(".//sec"):
            if sec.attrib.get("sec-type", "").lower() == "results":
                relevant = True
            else:
                title_el = sec.find("title")
                relevant = title_el is not None and "result" in _normalize_title(title_el.text)
            if not relevant:
                continue

            results_pmids.update(pub.text.strip() for pub in sec.findall('.//pub-id[@pub-id-type="pmid"]') if pub.text)
            for xref in sec.findall('.//xref[@ref-type="bibr"]'):
                rid = xref.attrib.get("rid", "")
                results_pmids.update(all_refs.get(rid, set()))

        return results_pmids, all_pmids

    def stream_jsonl(path: Path):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def _process_one(pmcid: str, xml_text: Optional[str], item: dict, fout):
        if xml_text:
            results_pmids, all_pmids = extract_reference_pmids_by_section(xml_text)
            item["results-section-pmids"] = sorted(results_pmids)
            item["references-pmids"] = sorted(all_pmids)
        else:
            item["results-section-pmids"] = []
            item["references-pmids"] = []

        # Only write to output if results-section-pmids is not empty
        if item["results-section-pmids"]:
            fout.write(json.dumps(item) + "\n")
        # If results-section-pmids is empty, skip this entry entirely

    # Main processing logic
    cache_dir = xml_dump_dir
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    items = list(stream_jsonl(Path(input_file)))
    item_by_pmc = {str(it.get("pmcid") or it.get("pmc-id") or "").strip(): it for it in items}
    all_pmcs = list(item_by_pmc.keys())

    cached_xml: Dict[str, str] = {}
    to_fetch_pmcs: List[str] = []
    for pmc in all_pmcs:
        if cache_dir and (cache_dir / f"{pmc}.xml").exists():
            try:
                cached_xml[pmc] = (cache_dir / f"{pmc}.xml").read_text(encoding="utf-8")
            except OSError:
                to_fetch_pmcs.append(pmc)
        else:
            to_fetch_pmcs.append(pmc)

    with open(output_file, "w", encoding="utf-8") as fout:
        pbar = tqdm(total=len(all_pmcs), desc="Processing records")

        for pmc, xml_text in cached_xml.items():
            _process_one(pmc, xml_text, item_by_pmc[pmc], fout)
            pbar.update(1)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(fetch_articles_xml_batch, batch, api_key): batch for batch in chunked(to_fetch_pmcs, batch_size)}
            for fut in as_completed(fut_map):
                batch_xmls = fut.result()
                for pmc in fut_map[fut]:
                    xml_text = batch_xmls.get(pmc)
                    if xml_text and cache_dir:
                        try:
                            (cache_dir / f"{pmc}.xml").write_text(xml_text, encoding="utf-8")
                        except OSError:
                            pass
                    _process_one(pmc, xml_text, item_by_pmc[pmc], fout)
                    pbar.update(1)
        pbar.close()

    print(f"✅ Step 4 complete: Processed {len(all_pmcs)} records → {output_file}")
    return output_file


def step_05_date_correction_retrieve(
    sr_file=None,
    all_collection_folder=ALL_COLLECTION_DIR,
    output_file=None,
    max_workers=12
):
    """
    Step 5: Date correction and document retrieval
    """
    if sr_file is None:
        sr_file = SR_MAPPING_DIR / "filtered_references.jsonl"
    if output_file is None:
        output_file = SR_AUGMENTED_DIR / "all.jsonl"

    print(f"📅 Step 5: Date correction and retrieval from {sr_file}")

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
                            d[pmid] = item.get("max-date", "")
                    except json.JSONDecodeError:
                        continue
            return d

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(load_file, files), total=len(files), desc="Loading all_collection"))
        for r in results:
            data.update(r)
        return data

    def get_first_and_last_valid_dates(ref_pmids, all_dates):
        """Return the first and last valid date from the reference PMIDs, shifted by +1 day."""
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

    def process_and_rewrite_reviews(sr_file, all_doc_dates, output_file):
        """Read one .jsonl SR file, augment with date range & retrieved IDs, and write all output to a single file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(sr_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
            for line in tqdm(fin, desc="Processing SR reviews"):
                try:
                    review = json.loads(line)
                    ref_pmids = review.get("references-pmids", [])
                    first_ref_date, last_ref_date = get_first_and_last_valid_dates(ref_pmids, all_doc_dates)
                    review["first_ref_date"] = first_ref_date
                    review["last_ref_date"] = last_ref_date

                    boolean_query = review.get("boolean_query", "")
                    if boolean_query and first_ref_date and last_ref_date:
                        retrieved_ids = retrieve_documents(boolean_query, (first_ref_date, last_ref_date))
                        print(f"Retrieved {len(retrieved_ids)} documents for query")
                        review["retrieved_ids"] = retrieved_ids

                    fout.write(json.dumps(review) + "\n")
                except json.JSONDecodeError:
                    continue

    print("Loading all collection document dates...")
    all_doc_dates = load_all_docs(all_collection_folder, max_workers)

    print("Processing and rewriting SR reviews...")
    process_and_rewrite_reviews(sr_file, all_doc_dates, output_file)

    print(f"✅ Step 5 complete: Augmented SR reviews written to {output_file}")
    return output_file