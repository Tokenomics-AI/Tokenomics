#!/usr/bin/env python3
"""Complete platform validation runner - orchestrates all test suites."""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class CompleteValidationRunner:
    """Orchestrates all validation test suites."""
    
    def __init__(self):
        self.results = {
            'validation_start': datetime.now().isoformat(),
            'ml_evaluation': None,
            'platform_integration': None,
            'component_validation': None,
            'errors': [],
            'summary': {}
        }
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "validation_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def run_ml_evaluation(self) -> bool:
        """Run ML Model Evaluation (Phase 1)."""
        print("\n" + "=" * 80)
        print("PHASE 1: ML MODEL EVALUATION")
        print("=" * 80)
        print()
        
        # Check if results already exist
        eval_results_file = self.project_root / "training_data" / "evaluation_report.json"
        if eval_results_file.exists():
            try:
                with open(eval_results_file, 'r', encoding='utf-8') as f:
                    self.results['ml_evaluation'] = json.load(f)
                print(f"✓ ML Evaluation results already exist: {eval_results_file}")
                print("  Skipping re-execution. Using existing results.")
                return True
            except Exception as e:
                print(f"⚠ Warning: Failed to load existing results: {e}")
                print("  Will re-run evaluation...")
        
        try:
            script_path = self.project_root / "scripts" / "evaluate_ml_models.py"
            
            print(f"Executing: {script_path}")
            print("This may take 1-2 hours (1,680 queries with 3s delay)...")
            print()
            
            # Run the evaluation script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.project_root),
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            # Load results from output file
            if eval_results_file.exists():
                with open(eval_results_file, 'r', encoding='utf-8') as f:
                    self.results['ml_evaluation'] = json.load(f)
                print(f"\n✓ ML Evaluation results loaded from: {eval_results_file}")
                return result.returncode == 0
            else:
                error_msg = "ML evaluation completed but results file not found"
                print(f"\n⚠ {error_msg}")
                self.results['errors'].append({
                    'phase': 'ml_evaluation',
                    'error': error_msg
                })
                return False
                
        except Exception as e:
            error_msg = f"ML evaluation failed: {str(e)}"
            print(f"\n✗ {error_msg}")
            self.results['errors'].append({
                'phase': 'ml_evaluation',
                'error': error_msg
            })
            return False
    
    def run_platform_integration(self) -> bool:
        """Run Comprehensive Platform Integration Test (Phase 2)."""
        print("\n" + "=" * 80)
        print("PHASE 2: COMPREHENSIVE PLATFORM INTEGRATION TEST")
        print("=" * 80)
        print()
        
        try:
            script_path = self.project_root / "tests" / "integration" / "test_comprehensive_platform.py"
            
            print(f"Executing: {script_path}")
            print("This may take 10-15 minutes...")
            print()
            
            # Run the integration test
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.project_root),
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            # Load results from output file
            platform_results_file = self.project_root / "tests" / "integration" / "comprehensive_test_results.json"
            if platform_results_file.exists():
                with open(platform_results_file, 'r', encoding='utf-8') as f:
                    self.results['platform_integration'] = json.load(f)
                print(f"\n✓ Platform Integration results loaded from: {platform_results_file}")
                return result.returncode == 0
            else:
                error_msg = "Platform integration test completed but results file not found"
                print(f"\n⚠ {error_msg}")
                self.results['errors'].append({
                    'phase': 'platform_integration',
                    'error': error_msg
                })
                return False
                
        except Exception as e:
            error_msg = f"Platform integration test failed: {str(e)}"
            print(f"\n✗ {error_msg}")
            self.results['errors'].append({
                'phase': 'platform_integration',
                'error': error_msg
            })
            return False
    
    def run_component_validation(self) -> bool:
        """Run Component-Level Validation (Phase 3)."""
        print("\n" + "=" * 80)
        print("PHASE 3: COMPONENT-LEVEL VALIDATION")
        print("=" * 80)
        print()
        
        try:
            script_path = self.project_root / "scripts" / "validate_components.py"
            
            print(f"Executing: {script_path}")
            print("This should take 1-2 minutes...")
            print()
            
            # Run the component validation
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.project_root),
                capture_output=True,  # Capture output to parse results
                text=True
            )
            
            # Load results from output file (created by enhanced validator)
            component_results_file = self.project_root / "training_data" / "component_validation_results.json"
            if component_results_file.exists():
                with open(component_results_file, 'r', encoding='utf-8') as f:
                    self.results['component_validation'] = json.load(f)
                print(f"\n✓ Component Validation results loaded from: {component_results_file}")
                return result.returncode == 0
            else:
                # If file doesn't exist, try to parse from output
                # This is a fallback if the validator hasn't been enhanced yet
                error_msg = "Component validation results file not found"
                print(f"\n⚠ {error_msg}")
                self.results['errors'].append({
                    'phase': 'component_validation',
                    'error': error_msg
                })
                # Still return True if the script ran successfully
                return result.returncode == 0
                
        except Exception as e:
            error_msg = f"Component validation failed: {str(e)}"
            print(f"\n✗ {error_msg}")
            self.results['errors'].append({
                'phase': 'component_validation',
                'error': error_msg
            })
            return False
    
    def generate_summary(self):
        """Generate summary statistics."""
        summary = {
            'phases_completed': 0,
            'phases_failed': 0,
            'total_errors': len(self.results['errors']),
            'ml_evaluation_status': 'not_run',
            'platform_integration_status': 'not_run',
            'component_validation_status': 'not_run',
        }
        
        if self.results['ml_evaluation'] is not None:
            summary['ml_evaluation_status'] = 'completed'
            summary['phases_completed'] += 1
        elif any(e['phase'] == 'ml_evaluation' for e in self.results['errors']):
            summary['ml_evaluation_status'] = 'failed'
            summary['phases_failed'] += 1
        
        if self.results['platform_integration'] is not None:
            summary['platform_integration_status'] = 'completed'
            summary['phases_completed'] += 1
        elif any(e['phase'] == 'platform_integration' for e in self.results['errors']):
            summary['platform_integration_status'] = 'failed'
            summary['phases_failed'] += 1
        
        if self.results['component_validation'] is not None:
            summary['component_validation_status'] = 'completed'
            summary['phases_completed'] += 1
        elif any(e['phase'] == 'component_validation' for e in self.results['errors']):
            summary['component_validation_status'] = 'failed'
            summary['phases_failed'] += 1
        
        self.results['summary'] = summary
        self.results['validation_end'] = datetime.now().isoformat()
    
    def save_results(self):
        """Save validation results to JSON file."""
        results_file = self.results_dir / "validation_run_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n✓ Validation run results saved to: {results_file}")
        return results_file
    
    def run_all(self):
        """Run all validation phases."""
        print("=" * 80)
        print("COMPLETE PLATFORM VALIDATION TEST SUITE")
        print("=" * 80)
        print(f"Started at: {self.results['validation_start']}")
        print()
        
        # Phase 1: ML Evaluation
        ml_success = self.run_ml_evaluation()
        
        # Phase 2: Platform Integration
        platform_success = self.run_platform_integration()
        
        # Phase 3: Component Validation
        component_success = self.run_component_validation()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        # Print summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"ML Evaluation: {self.results['summary']['ml_evaluation_status']}")
        print(f"Platform Integration: {self.results['summary']['platform_integration_status']}")
        print(f"Component Validation: {self.results['summary']['component_validation_status']}")
        print(f"\nPhases Completed: {self.results['summary']['phases_completed']}/3")
        print(f"Phases Failed: {self.results['summary']['phases_failed']}/3")
        print(f"Total Errors: {self.results['summary']['total_errors']}")
        print()
        
        if self.results['summary']['total_errors'] > 0:
            print("Errors encountered:")
            for error in self.results['errors']:
                print(f"  - {error['phase']}: {error['error']}")
            print()
        
        print("Next step: Run report generator:")
        print("  python3 scripts/generate_validation_report.py")
        print("=" * 80)
        
        return all([ml_success, platform_success, component_success])


if __name__ == "__main__":
    runner = CompleteValidationRunner()
    success = runner.run_all()
    sys.exit(0 if success else 1)

