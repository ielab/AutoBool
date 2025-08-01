"""
config.py - Centralized configuration for PubMed processing pipeline
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path("../data")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

# PubMed directories
PUBMED_RAW_DIR = RAW_DIR / "pubmed"
PUBMED_PROCESSED_DIR = PROCESSED_DIR / "pubmed"
SR_MAPPING_DIR = PUBMED_PROCESSED_DIR / "sr_mapping"
ALL_COLLECTION_DIR = PUBMED_PROCESSED_DIR / "all_collection"
SR_AUGMENTED_DIR = PUBMED_PROCESSED_DIR / "sr_augmented"

# File paths
PUBMED_IDS_FILE = BASE_DIR / "pubmed_ids_sr.txt"
CLEF_SEED_FILES = [
    RAW_DIR / "CLEF-SEED-topics" / "CLEF-2018_with_pmids.jsonl",
    RAW_DIR / "CLEF-SEED-topics" / "seed_collection_with_pmid.jsonl"
]

# Processing parameters
DEFAULT_QUERY = '"pubmed pmc open access"[filter] AND ("systematic review"[pt])'
PUBMED_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
PUBMED_START_INDEX = 363  # Files start from pubmed25n0001.xml.gz, not 0000
PUBMED_MAX_INDEX = 364

# Multiprocessing settings
DEFAULT_NUM_THREADS = 4
DEFAULT_MAX_WORKERS = 5
DEFAULT_BATCH_SIZE = 20

# API settings
DEFAULT_API_KEY = "api_key"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
HEADERS = {"User-Agent": "SR-Reference-Filter/1.0 (https://github.com/your-org)"}
TIMEOUT = 15
MAX_RETRIES = 3

# XML dump directory for caching
XML_DUMP_DIR = Path("xml_dump")

def ensure_directories():
    """Ensure all necessary directories exist"""
    directories = [
        BASE_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        PUBMED_RAW_DIR,
        PUBMED_PROCESSED_DIR,
        SR_MAPPING_DIR,
        ALL_COLLECTION_DIR,
        SR_AUGMENTED_DIR,
        XML_DUMP_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"✅ Ensured all directories exist")