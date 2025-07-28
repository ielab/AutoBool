import os
import json
from datasets import Dataset, DatasetDict
from typing import List
from tqdm import tqdm
from utils import process_and_push

sample_format = [{
    "content": "You are an expert systematic review information specialist.\nYou are tasked to formulate a systematic review Boolean query in response to a research topic. The final Boolean query must be enclosed within <answer> </answer> tags. Do not include any explanation or reasoning.",
    "role": "system"
}, {
    "content": 'You are given a systematic review research topic, with the topic title "{topic}".\n'
               "Your task is to formulate a highly effective Boolean query in MEDLINE format for PubMed.\n"
               "The query should balance **high recall** (capturing all relevant studies) with **reasonable precision** (avoiding irrelevant results):\n"
               "- Use both free-text terms and MeSH terms (e.g., chronic pain[tiab], Pain[mh]).\n"
               "- **Do not wrap terms or phrases in double quotes**, as this disables automatic term mapping (ATM).\n"
               "- Combine synonyms or related terms within a concept using OR.\n"
               "- Combine different concepts using AND.\n"
               "- Use wildcards (*) to capture word variants (e.g., vaccin* → vaccine, vaccination):\n"
               "  - Terms must have ≥4 characters before the * (e.g., colo*)\n"
               "  - Wildcards work with field tags (e.g., breastfeed*[tiab]).\n"
               "- Field tags limit the search to specific fields and disable ATM.\n"
               "- Do not include date limits.\n"
               "- Tag term using term field (e.g., covid-19[ti] vaccine[ti] children[ti]) when needed.\n"
               "**Only use the following allowed field tags:**\n"
               "Title: [ti], Abstract: [ab], Title/Abstract: [tiab]\n"
               "MeSH: [mh], Major MeSH: [majr], Supplementary Concept: [nm]\n"
               "Text Words: [tw], All Fields: [all]\n"
               "Publication Type: [pt], Language: [la]\n\n"
               "Output and only output the formulated Boolean query inside <answer></answer> tags. Do not include any explanation or content outside or inside the <answer> tags.",
    "role": "user"
}]


sample_out_format = [{
    "content": "<answer>{boolean}</answer>",
    "role": "assistant"
}]

# Example usage
if __name__ == "__main__":
    # process_and_push("../data/processed/pubmed/sr", "xxx/pubmed-pmc-oa-sr-dataset")
    # process_and_push("../data/processed/pubmed/ma", "xxx/pubmed-pmc-oa-ma-dataset")
    # process_and_push("../data/processed/pubmed/re", "xxx/pubmed-pmc-oa-re-dataset")
    # process_and_push("../data/processed/pubmed/sr_augmented", "xxx/pubmed-pmc-oa-sr-dataset-no-reasoning", completion=False, sample_format=sample_format)
    #process_and_push("../data/processed/clef_augmented", "xxx/clef-oa-sr-dataset-no-reasoning", completion=False, sample_format=sample_format, split=False)
    # process_and_push("../data/processed/pubmed/sr_augmented", "xxx/pubmed-pmc-oa-sr-dataset-no-reasoning-sft", completion=True, sample_format=sample_format, sample_out_format=sample_out_format)
    # process_and_push("../data/processed/clef_augmented",
    #                  "xxx/clef-oa-sr-dataset-no-reasoning-qwen-tokenized",
    #   completion=False, sample_format=sample_format, tokenizer_path="Qwen/Qwen3-0.6B", enable_thinking=False, split=False)
    #process_and_push("../data/processed/pubmed/sr_augmented", "xxx/pubmed-pmc-oa-sr-dataset-no-reasoning-qwen-tokenized",
                    # completion=False, sample_format=sample_format, tokenizer_path="Qwen/Qwen3-0.6B", enable_thinking=False, split=True)
    process_and_push("../data/processed/pubmed/sr_augmented_result", "xxx/pubmed-pmc-oa-sr-dataset-filtered-no-reasoning",
                     completion=False, sample_format=sample_format, split=True)
    process_and_push("../data/processed/pubmed/sr_augmented_result",
                     "xxx/pubmed-pmc-oa-sr-dataset-filtered-no-reasoning-qwen-tokenized",
                     completion=False, sample_format=sample_format, tokenizer_path="Qwen/Qwen3-0.6B", enable_thinking=False, split=True)

    # process_and_push("../data/processed/seed_augmented", "xxx/seed-oa-sr-dataset-no-reasoning", completion=False, sample_format=sample_format, split=False)
    # process_and_push("../data/processed/seed_augmented",
    #                  "xxx/seed-oa-sr-dataset-no-reasoning-qwen-tokenized", completion=False, sample_format=sample_format, tokenizer_path="Qwen/Qwen3-0.6B", enable_thinking=False, split=False)
    #
