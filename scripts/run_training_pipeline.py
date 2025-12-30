#!/usr/bin/env python3
"""Orchestrate the complete training pipeline."""

import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.training_config import TRAINING_CONFIG


def run_command(script_name: str, description: str, *args) -> bool:
    """Run a script and return success status."""
    print("\n" + "=" * 80)
    print(description)
    print("=" * 80)
    
    script_path = Path(__file__).parent / script_name
    cmd = ["python3", str(script_path)] + list(args)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Error running command: {e}")
        return False


def main():
    """Run the complete training pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description="Run complete ML training pipeline")
    parser.add_argument("--sample-size", type=int, help="Number of queries to run for data collection")
    parser.add_argument("--skip-preparation", action="store_true", help="Skip data preparation step")
    parser.add_argument("--skip-collection", action="store_true", help="Skip data collection step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation step")
    parser.add_argument("--max-eval-queries", type=int, help="Maximum test queries for evaluation")
    args = parser.parse_args()
    
    print("=" * 80)
    print("ML Training Pipeline")
    print("=" * 80)
    print()
    print("This pipeline will:")
    print("  1. Prepare dataset (clean, split 80:20)")
    print("  2. Collect training data (run queries through platform)")
    print("  3. Train ML models")
    print("  4. Evaluate on test set")
    print()
    
    results = {}
    
    # Step 1: Data Preparation
    if not args.skip_preparation:
        success = run_command(
            "prepare_customer_support_dataset.py",
            "Step 1: Data Preparation"
        )
        results["preparation"] = success
        if not success:
            print("\n✗ Data preparation failed. Stopping pipeline.")
            return 1
    else:
        print("\n⚠ Skipping data preparation (--skip-preparation)")
        results["preparation"] = True
    
    # Step 2: Data Collection
    if not args.skip_collection:
        collection_args = []
        if args.sample_size:
            collection_args.extend(["--sample-size", str(args.sample_size)])
        
        success = run_command(
            "collect_training_data.py",
            "Step 2: Data Collection",
            *collection_args
        )
        results["collection"] = success
        if not success:
            print("\n⚠ Data collection had errors, but continuing...")
    else:
        print("\n⚠ Skipping data collection (--skip-collection)")
        results["collection"] = True
    
    # Step 3: Model Training
    if not args.skip_training:
        success = run_command(
            "train_ml_models.py",
            "Step 3: Model Training"
        )
        results["training"] = success
        if not success:
            print("\n⚠ Model training had errors, but continuing...")
    else:
        print("\n⚠ Skipping model training (--skip-training)")
        results["training"] = True
    
    # Step 4: Evaluation
    if not args.skip_evaluation:
        eval_args = []
        if args.max_eval_queries:
            eval_args.extend(["--max-queries", str(args.max_eval_queries)])
        
        success = run_command(
            "evaluate_ml_models.py",
            "Step 4: Model Evaluation",
            *eval_args
        )
        results["evaluation"] = success
    else:
        print("\n⚠ Skipping evaluation (--skip-evaluation)")
        results["evaluation"] = True
    
    # Final summary
    print("\n" + "=" * 80)
    print("Pipeline Summary")
    print("=" * 80)
    
    for step, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {step.capitalize()}: {status}")
    
    all_passed = all(results.values())
    
    print()
    if all_passed:
        print("✓ Pipeline completed successfully!")
        print()
        print("Next steps:")
        print("  - Review evaluation reports in training_data/")
        print("  - Models are now trained and ready to use")
        print("  - Platform will use ML models instead of heuristics")
    else:
        print("⚠ Pipeline completed with some errors")
        print("  Review the output above for details")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())



