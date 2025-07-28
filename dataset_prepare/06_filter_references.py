#!/usr/bin/env python3
"""
filter_sr_references.py (v0.8 – **Cache‑aware, batched, multi‑threaded**)
=======================================================================

New in *v0.8*
-------------
1. **XML cache** – before hitting Entrez we look for ``xml_dump/PMCxxxxx.xml``.
   If it’s already on disk we parse it directly and skip the network call.
2. **Output resume** – unchanged from v0.7: if a PMCID already appears in the
   output file and you didn’t pass ``--overwrite``, we skip it entirely.
3. **Smarter batching** – we only batch‑fetch IDs that *aren’t* cached locally,
   so you never waste quota re‑downloading.

Everything else (thread pool, default paths, RID debug file, embedded key,
Results‑section parsing) stays the same.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from typing import Dict, List, Set

import requests
from tqdm import tqdm

# ────────────────────────────────────────────
# API / network constants
# ────────────────────────────────────────────
DEFAULT_API_KEY = "f1131df1583e6a5afdb564763a528dad9b08"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
HEADERS = {"User-Agent": "SR-Reference-Filter/0.8 (https://github.com/your-org)"}
TIMEOUT = 15
MAX_RETRIES = 3

# ────────────────────────────────────────────
# Defaults / CLI tunables
# ────────────────────────────────────────────
DEFAULT_INPUT = Path("../data/processed/pubmed/sr_augmented/all.jsonl")
DEFAULT_OUTPUT = Path("../data/processed/pubmed/sr_augmented/filtered_references.jsonl")
DEFAULT_XML_DUMP_DIR = Path("xml_dump")
BATCH_SIZE_DEFAULT = 20
MAX_WORKERS_DEFAULT = 5

# ────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────

def chunked(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def _normalize_title(text: str) -> str:
    return (text or "").strip().lower()


# ────────────────────────────────────────────
# Network: batched efetch
# ────────────────────────────────────────────

def fetch_articles_xml_batch(pmcids: List[str], api_key: str | None = None) -> Dict[str, str]:
    """Fetch many PMCs at once. Returns {pmcid: xml_text}."""
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


# ────────────────────────────────────────────
# XML → PMIDs
# ────────────────────────────────────────────

def extract_result_section_pmids(xml_text: str) -> Set[str]:
    pmids: Set[str] = set()
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return pmids

    results_secs = []
    for sec in root.findall(".//sec"):
        if sec.attrib.get("sec-type", "").lower() == "results":
            results_secs.append(sec)
            continue
        title_el = sec.find("title")
        if title_el is not None and "result" in _normalize_title(title_el.text):
            results_secs.append(sec)

    rids: Set[str] = set()
    for rs in results_secs:
        rids.update(x.attrib.get("rid", "") for x in rs.findall('.//xref[@ref-type="bibr"]'))
        for pub in rs.findall('.//pub-id[@pub-id-type="pmid"]'):
            if pub.text:
                pmids.add(pub.text.strip())

    for rid in filter(None, rids):
        ref_el = root.find(f".//ref[@id='{rid}']")
        if ref_el is None:
            continue
        for pub in ref_el.findall('.//pub-id[@pub-id-type="pmid"]'):
            if pub.text:
                pmids.add(pub.text.strip())
    return pmids


# ────────────────────────────────────────────
# Streaming JSONL
# ────────────────────────────────────────────

def stream_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Filter SR reference PMIDs to only those cited in Results sections, using cached XML when possible.")
    ap.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--api-key", "-k", default=os.getenv("NCBI_API_KEY", DEFAULT_API_KEY))
    ap.add_argument("--xml-dump-dir", type=Path, default=DEFAULT_XML_DUMP_DIR, help="Directory for XML cache; '' disables caching/fetch skip")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS_DEFAULT)
    ap.add_argument("--dump-rids", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Handle "" to disable caching
    cache_dir: Path | None = None if (not args.xml_dump_dir or str(args.xml_dump_dir) == "") else args.xml_dump_dir
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Output resume
    if args.output.exists() and not args.overwrite:
        out_mode = "a"
        processed_pmcs = {json.loads(l).get("pmc-id") for l in args.output.read_text(encoding="utf-8").splitlines() if l.strip()}
    else:
        out_mode = "w"
        processed_pmcs: Set[str] = set()

    items = [it for it in stream_jsonl(args.input) if str(it.get("pmc-id") or it.get("pmcid") or "").strip() not in processed_pmcs]
    item_by_pmc = {str(it.get("pmc-id") or it.get("pmcid") or "").strip(): it for it in items}
    all_pmcs = list(item_by_pmc.keys())

    # Split into cached vs to‑fetch
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

    # Prepare debug RID map if requested
    rid_fh = None
    if args.dump_rids:
        rid_fh = args.output.with_suffix(".ridmap.jsonl").open("w", encoding="utf-8")

    with args.output.open(out_mode, encoding="utf-8") as fout:
        pbar = tqdm(total=len(all_pmcs), desc="Processing records")

        # 1️⃣ Process cached XML first (no network)
        for pmc, xml_text in cached_xml.items():
            _process_one(pmc, xml_text, item_by_pmc[pmc], fout, rid_fh)
            pbar.update(1)

        # 2️⃣ Fetch missing ones in batches via thread pool
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            fut_map = {ex.submit(fetch_articles_xml_batch, batch, args.api_key): batch for batch in chunked(to_fetch_pmcs, args.batch_size)}
            for fut in as_completed(fut_map):
                batch_xmls = fut.result()
                for pmc in fut_map[fut]:
                    xml_text = batch_xmls.get(pmc)
                    if xml_text and cache_dir:
                        try:
                            (cache_dir / f"{pmc}.xml").write_text(xml_text, encoding="utf-8")
                        except OSError:
                            pass
                    _process_one(pmc, xml_text, item_by_pmc[pmc], fout, rid_fh)
                    pbar.update(1)
        pbar.close()

    if rid_fh:
        rid_fh.close()
    print(f"[DONE] Processed {len(all_pmcs)} records → {args.output}")


def _process_one(pmcid: str, xml_text: str | None, item: dict, fout, rid_fh):
    pmids = extract_result_section_pmids(xml_text) if xml_text else set()
    item["results-section-pmids"] = sorted(pmids)
    fout.write(json.dumps(item) + "\n")
    if rid_fh is not None:
        rid_fh.write(json.dumps({"pmcid": pmcid, "pmids": item["results-section-pmids"]}) + "\n")


if __name__ == "__main__":
    main()
