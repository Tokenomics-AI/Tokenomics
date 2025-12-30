#!/usr/bin/env python3
"""Continue validation after ML evaluation completes - runs phases 2 and 3."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_complete_validation import CompleteValidationRunner

if __name__ == "__main__":
    runner = CompleteValidationRunner()
    
    # Check if ML evaluation results exist
    eval_results_file = runner.project_root / "training_data" / "evaluation_report.json"
    if not eval_results_file.exists():
        print("⚠ ML Evaluation results not found.")
        print("Please wait for ML evaluation to complete first.")
        print("Check progress: tail -f validation_results/ml_evaluation.log")
        sys.exit(1)
    
    # Load ML evaluation results
    import json
    with open(eval_results_file, 'r', encoding='utf-8') as f:
        runner.results['ml_evaluation'] = json.load(f)
    print("✓ ML Evaluation results loaded (skipping Phase 1)")
    
    # Run remaining phases
    print("\nContinuing with remaining validation phases...\n")
    
    # Phase 2: Platform Integration
    platform_success = runner.run_platform_integration()
    
    # Phase 3: Component Validation
    component_success = runner.run_component_validation()
    
    # Generate summary
    runner.generate_summary()
    runner.save_results()
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"ML Evaluation: completed (from existing results)")
    print(f"Platform Integration: {runner.results['summary']['platform_integration_status']}")
    print(f"Component Validation: {runner.results['summary']['component_validation_status']}")
    print(f"\nPhases Completed: {runner.results['summary']['phases_completed']}/3")
    print()
    
    print("Next step: Generate unified report:")
    print("  python3 scripts/generate_validation_report.py")
    print("=" * 80)
    
    sys.exit(0 if all([platform_success, component_success]) else 1)
