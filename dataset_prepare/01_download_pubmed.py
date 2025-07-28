import os
import requests
import gzip
import shutil
from tqdm import tqdm

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
DEST_DIR = "../data/raw/pubmed"
START_INDEX = 0
MAX_INDEX = 1300  # large upper limit, we'll stop when no more files are found

os.makedirs(DEST_DIR, exist_ok=True)

def download_and_decompress(file_index):
    file_name = f"pubmed25n{file_index:04d}.xml.gz"
    xml_name = file_name[:-3]
    url = f"{BASE_URL}{file_name}"
    gz_path = os.path.join(DEST_DIR, file_name)
    xml_path = os.path.join(DEST_DIR, xml_name)

    # Skip if already decompressed
    if os.path.exists(xml_path):
        return True

    # Try to download the file
    try:
        response = requests.get(url, stream=True, timeout=200)
        if response.status_code == 200:
            with open(gz_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Decompress .gz to .xml
            with gzip.open(gz_path, 'rb') as f_in, open(xml_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

            os.remove(gz_path)  # Optional: remove .gz file after decompression
            return True
        else:
            return False  # File not found (likely end of available shards)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

if __name__ == "__main__":
    print("Starting download of PubMed baseline shards...")
    for i in tqdm(range(START_INDEX, MAX_INDEX + 1)):
        success = download_and_decompress(i)
        if not success:
            print(f"Stopping at index {i:04d}")
    print("Done.")