#!/usr/bin/env python3
"""Execute Phase 2 (Platform Integration Test) and document results."""

import sys
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests" / "integration"))

# Load environment
from dotenv import load_dotenv
load_dotenv(project_root / '.env')

def run_phase2_test():
    """Execute Phase 2 test and return results."""
    print("=" * 80)
    print("EXECUTING PHASE 2: COMPREHENSIVE PLATFORM INTEGRATION TEST")
    print("=" * 80)
    print()
    
    test_script = project_root / "tests" / "integration" / "test_comprehensive_platform.py"
    results_file = project_root / "tests" / "integration" / "comprehensive_test_results.json"
    
    print(f"Running: {test_script}")
    print("This may take 10-15 minutes...")
    print()
    
    try:
        # Option 1: Try importing and running directly
        try:
            # Import the test module
            import importlib.util
            spec = importlib.util.spec_from_file_location("test_comprehensive_platform", test_script)
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)
            
            # Run the test
            tester = test_module.ComprehensiveTest()
            test_results = tester.run_all_tests()
            
            # Save results (in case the test didn't)
            if not results_file.exists():
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(test_results, f, indent=2, default=str)
            
            # Load results
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    test_results = json.load(f)
                print(f"\n✓ Test results loaded from: {results_file}")
                success = test_results.get('summary', {}).get('failed', 1) == 0
                return test_results, success
            else:
                print(f"\n⚠ Results file not found: {results_file}")
                return test_results, True
                
        except Exception as import_error:
            print(f"Direct import failed: {import_error}")
            print("Falling back to subprocess execution...")
            
            # Option 2: Use subprocess
            result = subprocess.run(
                [sys.executable, str(test_script)],
                cwd=str(project_root),
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            # Load results if available
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    test_results = json.load(f)
                print(f"\n✓ Test results loaded from: {results_file}")
                success = result.returncode == 0
                return test_results, success
            else:
                print(f"\n⚠ Results file not found: {results_file}")
                return None, result.returncode == 0
            
    except Exception as e:
        print(f"\n✗ Error running Phase 2 test: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def document_results(test_results, success):
    """Document Phase 2 test results."""
    project_root = Path(__file__).parent.parent
    doc_file = project_root / "validation_results" / "PHASE2_RESULTS.md"
    
    if not test_results:
        documentation = f"""# Phase 2: Platform Integration Test Results

**Status**: ⚠️ Test execution completed but results file not found

**Execution Time**: {datetime.now().isoformat()}

## Summary

The Phase 2 test was executed, but the results file was not generated or could not be found.

## Next Steps

1. Check the test script output for errors
2. Verify the test script completed successfully
3. Check if `tests/integration/comprehensive_test_results.json` exists

"""
    else:
        summary = test_results.get('summary', {})
        tests = test_results.get('tests', [])
        errors = test_results.get('errors', [])
        
        # Organize tests by category
        test_categories = {}
        for test in tests:
            name = test.get('name', 'Unknown')
            # Extract category from test name
            if 'Environment' in name or 'Config' in name:
                category = 'Environment & Configuration'
            elif 'Memory' in name or 'Cache' in name:
                category = 'Memory Layer'
            elif 'Orchestrator' in name or 'Token' in name:
                category = 'Token Orchestrator'
            elif 'Bandit' in name:
                category = 'Bandit Optimizer'
            elif 'Integration' in name:
                category = 'Full Integration'
            elif 'Savings' in name:
                category = 'Component Savings'
            elif 'A/B' in name or 'Comparison' in name:
                category = 'A/B Comparison'
            elif 'Edge' in name:
                category = 'Edge Cases'
            else:
                category = 'Other'
            
            if category not in test_categories:
                test_categories[category] = []
            test_categories[category].append(test)
        
        # Build documentation
        documentation = f"""# Phase 2: Platform Integration Test Results

**Status**: {'✅ PASSED' if success and summary.get('failed', 0) == 0 else '⚠️ COMPLETED WITH ISSUES' if success else '❌ FAILED'}

**Execution Time**: {test_results.get('test_start', 'Unknown')} - {summary.get('test_end', 'Unknown')}

## Executive Summary

- **Total Tests**: {summary.get('total_tests', 0)}
- **Passed**: {summary.get('passed', 0)} ✅
- **Failed**: {summary.get('failed', 0)} ❌
- **Warnings**: {summary.get('warnings', 0)} ⚠️
- **Pass Rate**: {summary.get('pass_rate', 0)}%

"""
        
        if errors:
            documentation += f"""
## Errors Encountered

{len(errors)} error(s) found during testing:

"""
            for error in errors:
                documentation += f"- **{error.get('test', 'Unknown')}**: {error.get('error', 'No error message')}\n"
            documentation += "\n"
        
        # Detailed test results by category
        documentation += "## Detailed Test Results\n\n"
        
        for category, category_tests in test_categories.items():
            passed = sum(1 for t in category_tests if t.get('status') == 'pass')
            failed = sum(1 for t in category_tests if t.get('status') == 'fail')
            warnings = sum(1 for t in category_tests if t.get('status') == 'warning')
            
            status_icon = "✅" if failed == 0 else "⚠️" if warnings > 0 else "❌"
            documentation += f"### {status_icon} {category}\n\n"
            documentation += f"**Status**: {passed} passed, {failed} failed, {warnings} warnings\n\n"
            
            for test in category_tests:
                status = test.get('status', 'unknown')
                status_icon = "✅" if status == 'pass' else "❌" if status == 'fail' else "⚠️"
                name = test.get('name', 'Unknown Test')
                details = test.get('details', {})
                error = test.get('error')
                
                documentation += f"#### {status_icon} {name}\n\n"
                
                if details:
                    documentation += "**Details:**\n"
                    for key, value in details.items():
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value, indent=2)
                        documentation += f"- {key}: {value}\n"
                    documentation += "\n"
                
                if error:
                    documentation += f"**Error**: {error}\n\n"
        
        # Component Performance Analysis
        documentation += "\n## Component Performance Analysis\n\n"
        
        # Extract performance metrics from test details
        performance_metrics = {}
        for test in tests:
            details = test.get('details', {})
            if 'tokens_used' in details or 'cache_hit' in details or 'savings' in details:
                test_name = test.get('name', 'Unknown')
                performance_metrics[test_name] = details
        
        if performance_metrics:
            documentation += "### Key Performance Indicators\n\n"
            for test_name, metrics in performance_metrics.items():
                documentation += f"**{test_name}**\n"
                for key, value in metrics.items():
                    if key in ['tokens_used', 'tokens_saved', 'memory_savings', 'orchestrator_savings', 'bandit_savings']:
                        documentation += f"- {key}: {value}\n"
                documentation += "\n"
        
        # Recommendations
        documentation += "\n## Recommendations\n\n"
        
        if summary.get('failed', 0) > 0:
            documentation += "### Critical Issues\n"
            documentation += "The following tests failed and require immediate attention:\n\n"
            for test in tests:
                if test.get('status') == 'fail':
                    documentation += f"- **{test.get('name')}**: {test.get('error', 'No error message')}\n"
            documentation += "\n"
        
        if summary.get('warnings', 0) > 0:
            documentation += "### Warnings\n"
            documentation += "The following tests completed with warnings:\n\n"
            for test in tests:
                if test.get('status') == 'warning':
                    documentation += f"- **{test.get('name')}**: Review details for potential improvements\n"
            documentation += "\n"
        
        if summary.get('pass_rate', 0) == 100:
            documentation += "✅ **All tests passed!** The platform is functioning correctly.\n\n"
        elif summary.get('pass_rate', 0) >= 80:
            documentation += "⚠️ **Most tests passed.** Review failed tests and warnings for optimization opportunities.\n\n"
        else:
            documentation += "❌ **Multiple tests failed.** Review the detailed results above and address critical issues.\n\n"
    
    # Save documentation
    doc_file.parent.mkdir(exist_ok=True)
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write(documentation)
    
    print(f"\n✓ Phase 2 results documented in: {doc_file}")
    return doc_file

def main():
    """Main execution."""
    test_results, success = run_phase2_test()
    doc_file = document_results(test_results, success)
    
    print("\n" + "=" * 80)
    print("PHASE 2 EXECUTION COMPLETE")
    print("=" * 80)
    print(f"\nDocumentation saved to: {doc_file}")
    
    if test_results:
        summary = test_results.get('summary', {})
        print(f"\nTest Summary:")
        print(f"  Total Tests: {summary.get('total_tests', 0)}")
        print(f"  Passed: {summary.get('passed', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}")
        print(f"  Pass Rate: {summary.get('pass_rate', 0)}%")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

