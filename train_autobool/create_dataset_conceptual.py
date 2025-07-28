import os
import json
from datasets import Dataset, DatasetDict
from typing import List
from tqdm import tqdm
from utils import process_and_push

sample_format = [{
    "content": "You are an expert systematic review information specialist.\nFormulate a systematic review Boolean query using step-by-step reasoning inside <think> </think>, and output the final query inside <answer> </answer>.",
    "role": "system"
}, {
    "content": 'You are given a systematic review topic titled: "{topic}".\n'
               "Construct a Boolean query using the **conceptual method**, based on domain logic and structured thinking.\n\n"
               "**Step 1**: Identify 2–3 key concepts from the topic (e.g., Population, Intervention, Outcome).\n\n"
               "**Step 2**: For each concept:\n"
               "- List related terms: synonyms, variants, relevant MeSH terms.\n"
               "- Prioritise specific, high-precision terms.\n\n"
               "**Step 3**: Create a Boolean block per concept:\n"
               "- Combine terms using OR\n"
               "- Use free-text terms and MeSH terms (e.g., chronic pain[tiab], Pain[mh])\n"
               "- **Do not wrap terms or phrases in double quotes**, as this disables automatic term mapping (ATM)\n"
               "- Tag terms individually when needed (e.g., covid-19[ti] vaccine[ti] children[ti])\n"
               "- Field tags limit search scope and disable ATM\n\n"
               "**Step 4**: Use wildcards (*) to capture word variants (e.g., vaccin* → vaccine, vaccination):\n"
               "  - Terms must have ≥4 characters before the * (e.g., colo*)\n"
               "  - Wildcards work with field tags (e.g., breastfeed*[tiab]).\n\n"
               "**Step 5**: Combine all Boolean blocks using AND:\n"
               "((Concept1_term1[tiab] OR Concept1_term2[tiab] OR Concept1_termX[mh]) AND (Concept2_...))\n\n"
               "**Only use the following allowed field tags:**\n"
               "Title: [ti], Abstract: [ab], Title/Abstract: [tiab]\n"
               "MeSH: [mh], Major MeSH: [majr], Supplementary Concept: [nm]\n"
               "Text Words: [tw], All Fields: [all]\n"
               "Publication Type: [pt], Language: [la]\n\n"
               "Output your full reasoning inside <think>...</think>\n"
               "Output only the final Boolean query inside <answer>...</answer>\n"
               "Do not include any content outside these tags.\n"
               "Do not include date limits.",
    "role": "user"
}]





# Example usage
if __name__ == "__main__":
    process_and_push("../data/processed/pubmed/sr_augmented_result", "xxx/pubmed-pmc-oa-sr-dataset-filtered-reasoning-conceptual",
                     completion=False, sample_format=sample_format, split=True)

    # process_and_push("../data/processed/seed_augmented", "wshuai190/seed-oa-sr-dataset-reasoning-conceptual", completion=False, sample_format=sample_format, split=False)
