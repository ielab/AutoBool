#!/usr/bin/env python3
"""
main.py - Centralized PubMed processing pipeline orchestrator

This script runs all steps of the PubMed systematic review processing pipeline.
You can run all steps at once or specify individual steps to run.

Usage:
    python main.py                    # Run all steps
    python main.py --steps 0,1,2      # Run specific steps
    python main.py --start-from 3     # Start from step 3 onwards
    python main.py --help             # Show help
"""

import argparse
import sys
import time
from pathlib import Path

from config import ensure_directories
from steps import (
    step_00_get_pubmed_ids,
    step_01_download_pubmed,
    step_02_pmcid_date_mapping,
    step_03_remove_clef_seed_ids,
    step_04_get_references,
    step_05_date_correction_retrieve
)

# Add parent directory to Python path to import utils
import sys
sys.path.append('..')
from utils.logging_config import setup_dataset_logger

# Setup logger
logger = setup_dataset_logger("pubmed_pipeline")

STEP_FUNCTIONS = {
    0: {
        'func': step_00_get_pubmed_ids,
        'name': 'Get PubMed IDs',
        'description': 'Query PubMed API to get systematic review PMIDs'
    },
    1: {
        'func': step_01_download_pubmed,
        'name': 'Download PubMed Files',
        'description': 'Download and decompress PubMed baseline XML files'
    },
    2: {
        'func': step_02_pmcid_date_mapping,
        'name': 'PMC ID & Date Mapping',
        'description': 'Extract PMC IDs and publication dates from XML files'
    },
    3: {
        'func': step_03_remove_clef_seed_ids,
        'name': 'Remove CLEF Seed IDs',
        'description': 'Filter out CLEF seed collection PMIDs from systematic reviews'
    },
    4: {
        'func': step_04_get_references,
        'name': 'Extract References',
        'description': 'Extract reference PMIDs from systematic review articles'
    },
    5: {
        'func': step_05_date_correction_retrieve,
        'name': 'Date Correction & Retrieval',
        'description': 'Apply date corrections and retrieve related documents'
    }
}


def print_banner():
    """Print a nice banner"""
    logger.info("=" * 70)
    logger.info("🔬 PubMed Systematic Review Processing Pipeline")
    logger.info("=" * 70)


def print_step_summary():
    """Print summary of all available steps"""
    logger.info("\n📋 Available Steps:")
    logger.info("-" * 50)
    for step_num, step_info in STEP_FUNCTIONS.items():
        logger.info(f"  Step {step_num}: {step_info['name']}")
        logger.info(f"           {step_info['description']}")
    logger.info("-" * 50)


def run_step(step_num, verbose=True):
    """Run a single step"""
    if step_num not in STEP_FUNCTIONS:
        logger.error(f"Step {step_num} does not exist")
        return False

    step_info = STEP_FUNCTIONS[step_num]

    if verbose:
        logger.info(f"\n🚀 Starting Step {step_num}: {step_info['name']}")
        logger.info(f"   {step_info['description']}")
        logger.info("-" * 50)

    start_time = time.time()

    try:
        result = step_info['func']()
        end_time = time.time()
        duration = end_time - start_time

        if verbose:
            print(f"✅ Step {step_num} completed successfully in {duration:.1f}s")
            if result:
                print(f"   Output: {result}")

        return True

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"❌ Step {step_num} failed after {duration:.1f}s")
        print(f"   Error: {str(e)}")

        if verbose:
            import traceback
            print(f"   Traceback:")
            traceback.print_exc()

        return False


def run_pipeline(steps_to_run=None, start_from=None, verbose=True):
    """Run the complete pipeline or specified steps"""

    if steps_to_run is None and start_from is None:
        # Run all steps
        steps_to_run = list(STEP_FUNCTIONS.keys())
    elif start_from is not None:
        # Run from specific step onwards
        steps_to_run = [i for i in STEP_FUNCTIONS.keys() if i >= start_from]

    if verbose:
        print(f"\n🎯 Running steps: {steps_to_run}")

    total_start_time = time.time()
    successful_steps = []
    failed_steps = []

    for step_num in steps_to_run:
        success = run_step(step_num, verbose)

        if success:
            successful_steps.append(step_num)
        else:
            failed_steps.append(step_num)

            # Ask if user wants to continue on failure
            if verbose and len(steps_to_run) > 1:
                response = input(f"\n⚠️  Step {step_num} failed. Continue with remaining steps? (y/n): ")
                if response.lower() != 'y':
                    break

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Print summary
    print("\n" + "=" * 70)
    print("📊 PIPELINE SUMMARY")
    print("=" * 70)
    print(f"⏱️  Total runtime: {total_duration:.1f}s ({total_duration / 60:.1f} minutes)")
    print(f"✅ Successful steps: {len(successful_steps)}")
    print(f"❌ Failed steps: {len(failed_steps)}")

    if successful_steps:
        print(f"\n✅ Completed steps: {successful_steps}")

    if failed_steps:
        print(f"\n❌ Failed steps: {failed_steps}")

    print("=" * 70)

    return len(failed_steps) == 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PubMed Systematic Review Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run all steps
  python main.py --steps 0,1,2      # Run steps 0, 1, and 2
  python main.py --start-from 3     # Run steps 3, 4, 5
  python main.py --list-steps       # Show available steps
  python main.py --quiet            # Run with minimal output
        """
    )

    parser.add_argument(
        '--steps',
        type=str,
        help='Comma-separated list of steps to run (e.g., "0,1,2")'
    )

    parser.add_argument(
        '--start-from',
        type=int,
        help='Start from this step and run all subsequent steps'
    )

    parser.add_argument(
        '--list-steps',
        action='store_true',
        help='List all available steps and exit'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Run with minimal output'
    )

    parser.add_argument(
        '--setup-only',
        action='store_true',
        help='Only create directories and exit'
    )

    args = parser.parse_args()

    # Always print banner unless quiet
    if not args.quiet:
        print_banner()

    # Setup directories
    if not args.quiet:
        print("\n🔧 Setting up directories...")
    ensure_directories()

    if args.setup_only:
        print("✅ Directory setup complete. Exiting.")
        return 0

    # List steps if requested
    if args.list_steps:
        print_step_summary()
        return 0

    # Parse steps to run
    steps_to_run = None
    if args.steps:
        try:
            steps_to_run = [int(s.strip()) for s in args.steps.split(',')]
            # Validate steps
            invalid_steps = [s for s in steps_to_run if s not in STEP_FUNCTIONS]
            if invalid_steps:
                print(f"❌ Error: Invalid steps: {invalid_steps}")
                print(f"   Available steps: {list(STEP_FUNCTIONS.keys())}")
                return 1
        except ValueError:
            print("❌ Error: Invalid steps format. Use comma-separated integers (e.g., '0,1,2')")
            return 1

    if args.start_from is not None and args.start_from not in STEP_FUNCTIONS:
        print(f"❌ Error: Invalid start step: {args.start_from}")
        print(f"   Available steps: {list(STEP_FUNCTIONS.keys())}")
        return 1

    # Run the pipeline
    success = run_pipeline(
        steps_to_run=steps_to_run,
        start_from=args.start_from,
        verbose=not args.quiet
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())