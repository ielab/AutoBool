#!/usr/bin/env python3
"""
Unified Dataset Creation Script for AutoBool

This script consolidates the four separate create_dataset_*.py files into a single,
configurable script that can generate datasets with different prompt types.

Usage:
    python create_unified_dataset.py --prompt-type no_reason --data-path ../data/processed/pubmed/sr_augmented_result --hf-name your/dataset-name
    python create_unified_dataset.py --prompt-type reasoning --data-path ../data/processed/clef_augmented --hf-name your/clef-dataset --no-split
"""

import argparse
import os
import json
import sys
from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# Add utils to path for logging
sys.path.append('..')
from utils.logging_config import get_logger

# Import the core functions from utils.py 
from utils import process_and_push, build_prompt, create_split_dataset, load_jsonl_dataset

# Setup logger
logger = get_logger("autobool.dataset.unified")


class PromptTemplates:
    """Centralized prompt templates for different dataset types."""
    
    @staticmethod
    def get_no_reason_template() -> List[Dict]:
        """Template without reasoning - direct answer only."""
        return [{
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
    
    @staticmethod
    def get_reasoning_template() -> List[Dict]:
        """Template with explicit reasoning process."""
        return [{
            "content": "You are an expert systematic review information specialist.\nYou are tasked to formulate a systematic review Boolean query in response to a research topic.\nYour reasoning process should be enclosed within <think></think>, and the final Boolean query must be enclosed within <answer></answer> tags. Do not include anything outside of these tags.",
            "role": "system"
        }, {
            "content": 'You are given a systematic review research topic, with the topic title "{topic}".\n'
                       "Your task is to generate a highly effective Boolean query in MEDLINE format for PubMed.\n"
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
                       "- Tag terms using appropriate fields (e.g., covid-19[ti] vaccine[ti] children[ti]) when needed.\n"
                       "**Only use the following allowed field tags:**\n"
                       "Title: [ti], Abstract: [ab], Title/Abstract: [tiab]\n"
                       "MeSH: [mh], Major MeSH: [majr], Supplementary Concept: [nm]\n"
                       "Text Words: [tw], All Fields: [all]\n"
                       "Publication Type: [pt], Language: [la]\n\n"
                       "Output your full reasoning inside <think></think>.\n"
                       "Output the final Boolean query inside <answer></answer>.\n"
                       "Do not include any content outside these tags.",
            "role": "user"
        }]
    
    @staticmethod
    def get_conceptual_template() -> List[Dict]:
        """Template using conceptual method with structured thinking."""
        return [{
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
    
    @staticmethod
    def get_objective_template() -> List[Dict]:
        """Template using objective method with simulated examples."""
        return [{
            "content": "You are an expert systematic review information specialist.\nYou are tasked to formulate a systematic review Boolean query step by step as a reasoning process within <think> </think>, and provide the Boolean query formulated <answer> </answer>.",
            "role": "system"
        }, {
            "content": 'You are given a systematic review research topic, with the topic title "{topic}".\n'
                       "You need to simulate a Boolean query construction process using the **objective method**, which is grounded in domain expertise and structured logic.\n\n"
                       "**Step 1**: Simulate a concise title and abstract (2–3 sentences) of a *relevant and focused* article clearly aligned with the topic. This is a hypothetical but plausible example.\n\n"
                       "**Step 2**: Based on the simulated text, identify *key informative terms or phrases* that best represent the article's core concepts. Prioritise specificity and informativeness. Avoid overly broad or ambiguous terms.\n\n"
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
    
    @staticmethod
    def get_template(prompt_type: str) -> List[Dict]:
        """Get template by type name."""
        templates = {
            "no_reason": PromptTemplates.get_no_reason_template(),
            "reasoning": PromptTemplates.get_reasoning_template(),
            "conceptual": PromptTemplates.get_conceptual_template(),
            "objective": PromptTemplates.get_objective_template(),
        }
        
        if prompt_type not in templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}. Available: {list(templates.keys())}")
        
        return templates[prompt_type]


def create_unified_dataset(
    data_path: str,
    hf_name: str,
    prompt_type: str,
    seed: int = 42,
    completion: bool = False,
    enable_thinking: bool = True,
    tokenizer_path: Optional[str] = None,
    split: bool = True,
    sample_output_format: Optional[List[Dict]] = None
) -> None:
    """
    Create dataset with specified prompt type.
    
    Args:
        data_path: Path to input data directory
        hf_name: HuggingFace dataset name for upload
        prompt_type: Type of prompt template (no_reason, reasoning, conceptual, objective)
        seed: Random seed for data splitting
        completion: Whether to include completion format
        enable_thinking: Enable thinking tags for tokenizer
        tokenizer_path: Path to tokenizer (required if enable_thinking=False)
        split: Whether to create train/test/temporal splits
        sample_output_format: Output format template
    """
    logger.info(f"Creating unified dataset with prompt type: {prompt_type}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"HuggingFace name: {hf_name}")
    logger.info(f"Split dataset: {split}")
    
    # Get prompt template
    try:
        sample_format = PromptTemplates.get_template(prompt_type)
        logger.info(f"Using {prompt_type} prompt template")
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Use the existing process_and_push function
    try:
        process_and_push(
            folder_path=data_path,
            hf_name=hf_name,
            seed=seed,
            completion=completion,
            sample_format=sample_format,
            sample_out_format=sample_output_format,
            enable_thinking=enable_thinking,
            tokenizer_path=tokenizer_path,
            split=split
        )
        logger.info(f"Successfully created and uploaded dataset: {hf_name}")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Unified dataset creation script for AutoBool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create no-reasoning dataset with splits
  python create_unified_dataset.py --prompt-type no_reason \\
    --data-path ../data/processed/pubmed/sr_augmented_result \\
    --hf-name your-username/dataset-no-reason

  # Create reasoning dataset without splits
  python create_unified_dataset.py --prompt-type reasoning \\
    --data-path ../data/processed/clef_augmented \\
    --hf-name your-username/clef-reasoning --no-split

  # Create conceptual dataset with specific tokenizer
  python create_unified_dataset.py --prompt-type conceptual \\
    --data-path ../data/processed/seed_augmented \\
    --hf-name your-username/seed-conceptual \\
    --tokenizer-path Qwen/Qwen3-0.6B --no-thinking
        """
    )
    
    parser.add_argument(
        "--prompt-type", 
        type=str, 
        required=True,
        choices=["no_reason", "reasoning", "conceptual", "objective"],
        help="Type of prompt template to use"
    )
    
    parser.add_argument(
        "--data-path", 
        type=str, 
        required=True,
        help="Path to input data directory containing JSONL files"
    )
    
    parser.add_argument(
        "--hf-name", 
        type=str, 
        required=True,
        help="HuggingFace dataset name (e.g., 'username/dataset-name')"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for data splitting (default: 42)"
    )
    
    parser.add_argument(
        "--completion", 
        action="store_true",
        help="Include completion format in dataset"
    )
    
    parser.add_argument(
        "--no-thinking", 
        action="store_true",
        help="Disable thinking tags (requires --tokenizer-path)"
    )
    
    parser.add_argument(
        "--tokenizer-path", 
        type=str,
        help="Path to tokenizer (required when --no-thinking is used)"
    )
    
    parser.add_argument(
        "--no-split", 
        action="store_true",
        help="Don't create train/test/temporal splits, upload as single dataset"
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.no_thinking and not args.tokenizer_path:
        parser.error("--tokenizer-path is required when --no-thinking is used")
    
    if not os.path.exists(args.data_path):
        logger.error(f"Data path does not exist: {args.data_path}")
        return 1
    
    # Create dataset
    try:
        create_unified_dataset(
            data_path=args.data_path,
            hf_name=args.hf_name,
            prompt_type=args.prompt_type,
            seed=args.seed,
            completion=args.completion,
            enable_thinking=not args.no_thinking,
            tokenizer_path=args.tokenizer_path,
            split=not args.no_split
        )
        logger.info("Dataset creation completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())