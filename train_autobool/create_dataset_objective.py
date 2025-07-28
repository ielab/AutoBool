import os
import json
from datasets import Dataset, DatasetDict
from typing import List
from tqdm import tqdm
from utils import process_and_push

sample_format = [{
    "content": "You are an expert systematic review information specialist.\nYou are tasked to formulate a systematic review Boolean query step by step as a reasoning process within <think> </think>, and provide the Boolean query formulated <answer> </answer>.",
    "role": "system"
}, {
    "content": 'You are given a systematic review research topic, with the topic title "{topic}".\n'
               "You need to simulate a Boolean query construction process using the **objective method**, which is grounded in domain expertise and structured logic.\n\n"
               "**Step 1**: Simulate a concise title and abstract (2–3 sentences) of a *relevant and focused* article clearly aligned with the topic. This is a hypothetical but plausible example.\n\n"
               "**Step 2**: Based on the simulated text, identify *key informative terms or phrases* that best represent the article’s core concepts. Prioritise specificity and informativeness. Avoid overly broad or ambiguous terms.\n\n"
               "**Step 3**: Categorise each term into one of the following:\n"
               "- (A) Health conditions or populations (e.g., diabetes, adolescents)\n"
               "- (B) Treatments, interventions, or exposures (e.g., insulin therapy, air pollution)\n"
               "- (C) Study designs or methodologies (e.g., randomized controlled trial, cohort study)\n"
               "- (N/A) Not applicable to any of the above categories\n\n"
               "**Step 4**: Using the categorised terms, build a Boolean query in MEDLINE format for PubMed:\n"
               "- Combine synonyms or related terms within each category using OR\n"
               "- Use both free-text terms and MeSH terms (e.g., chronic pain[tiab], Pain[mh])\n"
               "- **Do not wrap terms or phrases in double quotes**, as this disables automatic term mapping (ATM)\n"
               "- Tag each term individually when needed (e.g., covid-19[ti] vaccine[ti] children[ti])\n"
               "- Field tags limit the search to specific fields and disable ATM\n\n"
               "**Step 5**: Use wildcards (*) to capture word variants (e.g., vaccin* → vaccine, vaccination):\n"
               "  - Terms must have ≥4 characters before the * (e.g., colo*)\n"
               "  - Wildcards work with field tags (e.g., breastfeed*[tiab]).\n\n"
               "**Step 6**: Combine all category blocks using AND:\n"
               "((itemA1[tiab] OR itemA2[tiab] OR itemA3[mh]) AND (itemB1[tiab] OR ...) AND (itemC1[tiab] OR ...))\n\n"
               "**Only use the following allowed field tags:**\n"
               "Title: [ti], Abstract: [ab], Title/Abstract: [tiab]\n"
               "MeSH: [mh], Major MeSH: [majr], Supplementary Concept: [nm]\n"
               "Text Words: [tw], All Fields: [all]\n"
               "Publication Type: [pt], Language: [la]\n\n"
               "Place your full reasoning (including simulated abstract, term list, classification, and query construction) inside <think></think>.\n"
               "Output the final Boolean query inside <answer></answer>.\n"
               "Do not include anything outside the <think> and <answer> tags.\n"
               "Do not include date restrictions.",
    "role": "user"
}]



# Example usage
if __name__ == "__main__":
    # process_and_push("../data/processed/pubmed/sr", "xxx/pubmed-pmc-oa-sr-dataset-reasoning-objective")
    # process_and_push("../data/processed/pubmed/ma", "xxx/pubmed-pmc-oa-ma-dataset-reasoning-objective")
    # process_and_push("../data/processed/pubmed/re", "xxx/pubmed-pmc-oa-re-dataset-reasoning-objective")
    # process_and_push("../data/processed/pubmed/sr_augmented", "xxx/pubmed-pmc-oa-sr-dataset-reasoning-objective", completion=False, sample_format=sample_format)
    # process_and_push("../data/processed/clef_augmented", "xxx/clef-oa-sr-dataset-reasoning-objective", completion=False, sample_format=sample_format, split=False)
    #process_and_push("../data/processed/seed_augmented", "xxx/seed-oa-sr-dataset-reasoning-objective", completion=False, sample_format=sample_format, split=False)
    process_and_push("../data/processed/pubmed/sr_augmented_result", "xxx/pubmed-pmc-oa-sr-dataset-filtered-reasoning-objective",
                     completion=False, sample_format=sample_format, split=True)
