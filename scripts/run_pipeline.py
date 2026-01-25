#!/usr/bin/env python3
"""
run_pipeline.py - Master script to run the complete analysis pipeline.

Usage: python scripts/run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

PIPELINE_STEPS = [
    ('01_data_preparation.py', 'Data Preparation'),
    ('02_univariate_cox.py', 'Univariate Cox Screening'),
    ('03_multivariate_cox.py', 'Multivariate Cox Modeling'),
    ('04_nomogram.py', 'Nomogram Generation'),
    ('05_internal_validation.py', 'Internal Validation'),
    ('06_external_validation.py', 'External Validation'),
    ('07_generate_figures.py', 'Generate Figures'),
]


def run_step(script_name, description):
    """Run a single pipeline step."""
    print(f"\n{'#' * 70}")
    print(f"# STEP: {description}")
    print(f"# Script: {script_name}")
    print(f"{'#' * 70}\n")

    script_path = SCRIPTS_DIR / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=SCRIPTS_DIR.parent,
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed!")
        return False

    print(f"\n✓ {description} completed successfully")
    return True


def main():
    print("=" * 70)
    print("ACC SURVIVAL ANALYSIS PIPELINE")
    print("Using Separate T, N, M Staging Components")
    print("=" * 70)

    success_count = 0
    for script, description in PIPELINE_STEPS:
        if run_step(script, description):
            success_count += 1
        else:
            print(f"\nPipeline stopped at: {description}")
            break

    print(f"\n{'=' * 70}")
    print(f"PIPELINE SUMMARY: {success_count}/{len(PIPELINE_STEPS)} steps completed")
    print("=" * 70)

    if success_count == len(PIPELINE_STEPS):
        print("\n✓ All pipeline steps completed successfully!")
        print("\nOutput structure:")
        print("  outputs/figures/  - All visualization figures")
        print("  outputs/tables/   - Summary tables and results")
        print("  outputs/models/   - Fitted models and specifications")
        print("  outputs/data/     - Processed datasets")
    else:
        print("\n❌ Pipeline incomplete. Check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
