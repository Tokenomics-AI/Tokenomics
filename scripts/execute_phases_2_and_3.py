#!/usr/bin/env python3
"""Execute Phase 2 (Platform Integration Test) and Phase 3 (Component Validation) with documentation."""

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


def check_prerequisites():
    """Check prerequisites before running tests."""
    print("=" * 80)
    print("CHECKING PREREQUISITES")
    print("=" * 80)
    print()
    
    # Check ML evaluation results (Phase 1)
    eval_results_file = project_root / "training_data" / "evaluation_report.json"
    if not eval_results_file.exists():
        print("⚠️  Warning: ML Evaluation results not found.")
        print("   Phase 1 results are recommended but not required for Phases 2 and 3.")
        print()
    else:
        print("✓ ML Evaluation results found (Phase 1 complete)")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment.")
        print("   Phase 2 requires API access. Phase 3 does not.")
        print()
    else:
        print("✓ API key found")
    
    print()


def run_phase2():
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


def run_phase3():
    """Execute Phase 3 test and return results."""
    print("\n" + "=" * 80)
    print("EXECUTING PHASE 3: COMPONENT-LEVEL VALIDATION")
    print("=" * 80)
    print()
    
    validation_script = project_root / "scripts" / "validate_components.py"
    results_file = project_root / "training_data" / "component_validation_results.json"
    
    print(f"Running: {validation_script}")
    print("This should take 1-2 minutes...")
    print()
    
    try:
        # Run the component validation
        result = subprocess.run(
            [sys.executable, str(validation_script)],
            cwd=str(project_root),
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        # Load results if available
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                validation_results = json.load(f)
            print(f"\n✓ Validation results loaded from: {results_file}")
            success = result.returncode == 0
            return validation_results, success
        else:
            print(f"\n⚠ Results file not found: {results_file}")
            return None, result.returncode == 0
            
    except Exception as e:
        print(f"\n✗ Error running Phase 3 validation: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def document_phase2(test_results, success):
    """Document Phase 2 test results."""
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


def document_phase3(validation_results, success):
    """Document Phase 3 validation results."""
    doc_file = project_root / "validation_results" / "PHASE3_RESULTS.md"
    
    if not validation_results:
        documentation = f"""# Phase 3: Component-Level Validation Results

**Status**: ⚠️ Validation execution completed but results file not found

**Execution Time**: {datetime.now().isoformat()}

## Summary

The Phase 3 validation was executed, but the results file was not generated or could not be found.

## Next Steps

1. Check the validation script output for errors
2. Verify the validation script completed successfully
3. Check if `training_data/component_validation_results.json` exists

"""
    else:
        # Extract results and test details (note: validate_components.py uses 'test_results' not 'results')
        results = validation_results.get('test_results', validation_results.get('results', {}))
        test_details = validation_results.get('test_details', {})
        execution_time = validation_results.get('validation_timestamp', validation_results.get('execution_time', datetime.now().isoformat()))
        
        # Calculate summary
        total_tests = len(results)
        passed = sum(1 for v in results.values() if v)
        failed = total_tests - passed
        pass_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        # Build documentation
        documentation = f"""# Phase 3: Component-Level Validation Results

**Status**: {'✅ PASSED' if success and failed == 0 else '⚠️ COMPLETED WITH ISSUES' if success else '❌ FAILED'}

**Execution Time**: {execution_time}

## Executive Summary

- **Total Tests**: {total_tests}
- **Passed**: {passed} ✅
- **Failed**: {failed} ❌
- **Pass Rate**: {pass_rate:.1f}%

## Component Test Results

"""
        
        # Component mapping
        component_names = {
            'memory_cache': 'Memory Cache Logic',
            'orchestrator': 'Orchestrator Logic',
            'bandit': 'Bandit Logic',
            'integration': 'Integration Logic',
            'token_counting': 'Token Counting',
            'bandit_algorithms': 'Bandit Algorithms'
        }
        
        for component_key, component_name in component_names.items():
            status = results.get(component_key, False)
            status_icon = "✅" if status else "❌"
            documentation += f"### {status_icon} {component_name}\n\n"
            
            # Get detailed test information
            if component_key in test_details:
                details = test_details[component_key]
                test_status = details.get('status', 'UNKNOWN')
                tests = details.get('tests', [])
                
                documentation += f"**Status**: {test_status}\n\n"
                
                if tests:
                    documentation += "**Test Details:**\n\n"
                    for test in tests:
                        test_name = test.get('name', 'Unknown')
                        test_status = test.get('status', 'UNKNOWN')
                        test_icon = "✅" if test_status == 'PASS' else "❌"
                        documentation += f"- {test_icon} {test_name}\n"
                        
                        if 'error' in test:
                            documentation += f"  - Error: {test['error']}\n"
                    documentation += "\n"
                
                if 'error' in details:
                    documentation += f"**Error**: {details['error']}\n\n"
                
                # Additional metrics if available
                for key, value in details.items():
                    if key not in ['test_name', 'status', 'tests', 'error']:
                        documentation += f"- {key}: {value}\n"
                documentation += "\n"
            else:
                documentation += f"**Status**: {'PASS' if status else 'FAIL'}\n\n"
        
        # Recommendations
        documentation += "\n## Recommendations\n\n"
        
        if failed > 0:
            documentation += "### Critical Issues\n"
            documentation += "The following component tests failed and require immediate attention:\n\n"
            for component_key, component_name in component_names.items():
                if not results.get(component_key, False):
                    error_msg = "Component validation failed"
                    if component_key in test_details and 'error' in test_details[component_key]:
                        error_msg = test_details[component_key]['error']
                    documentation += f"- **{component_name}**: {error_msg}\n"
            documentation += "\n"
        
        if pass_rate == 100:
            documentation += "✅ **All component tests passed!** The component logic is functioning correctly.\n\n"
        elif pass_rate >= 80:
            documentation += "⚠️ **Most component tests passed.** Review failed tests for optimization opportunities.\n\n"
        else:
            documentation += "❌ **Multiple component tests failed.** Review the detailed results above and address critical issues.\n\n"
    
    # Save documentation
    doc_file.parent.mkdir(exist_ok=True)
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write(documentation)
    
    print(f"\n✓ Phase 3 results documented in: {doc_file}")
    return doc_file


def create_combined_summary(phase2_results, phase2_success, phase3_results, phase3_success):
    """Create a combined summary document for both phases."""
    doc_file = project_root / "validation_results" / "PHASES_2_AND_3_SUMMARY.md"
    
    # Extract summaries
    phase2_summary = phase2_results.get('summary', {}) if phase2_results else {}
    phase3_results_dict = phase3_results.get('test_results', phase3_results.get('results', {})) if phase3_results else {}
    phase3_passed = sum(1 for v in phase3_results_dict.values() if v) if phase3_results else 0
    phase3_total = len(phase3_results_dict) if phase3_results else 0
    
    # Build combined summary
    documentation = f"""# Phases 2 and 3: Combined Validation Summary

**Generated**: {datetime.now().isoformat()}

## Overview

This document provides a high-level summary of both Phase 2 (Platform Integration Test) and Phase 3 (Component Validation) execution results.

## Phase 2: Platform Integration Test

**Status**: {'✅ PASSED' if phase2_success and phase2_summary.get('failed', 0) == 0 else '⚠️ COMPLETED WITH ISSUES' if phase2_success else '❌ FAILED'}

- **Total Tests**: {phase2_summary.get('total_tests', 0)}
- **Passed**: {phase2_summary.get('passed', 0)}
- **Failed**: {phase2_summary.get('failed', 0)}
- **Warnings**: {phase2_summary.get('warnings', 0)}
- **Pass Rate**: {phase2_summary.get('pass_rate', 0)}%

**Details**: See [PHASE2_RESULTS.md](PHASE2_RESULTS.md) for comprehensive test results.

## Phase 3: Component-Level Validation

**Status**: {'✅ PASSED' if phase3_success and phase3_passed == phase3_total else '⚠️ COMPLETED WITH ISSUES' if phase3_success else '❌ FAILED'}

- **Total Tests**: {phase3_total}
- **Passed**: {phase3_passed}
- **Failed**: {phase3_total - phase3_passed}
- **Pass Rate**: {(phase3_passed / phase3_total * 100) if phase3_total > 0 else 0:.1f}%

**Details**: See [PHASE3_RESULTS.md](PHASE3_RESULTS.md) for comprehensive validation results.

## Combined Metrics

### Overall Status

"""
    
    # Determine overall status
    phase2_ok = phase2_success and phase2_summary.get('failed', 0) == 0
    phase3_ok = phase3_success and phase3_passed == phase3_total
    
    if phase2_ok and phase3_ok:
        documentation += "✅ **Both phases passed successfully!**\n\n"
    elif phase2_ok or phase3_ok:
        documentation += "⚠️ **One phase completed with issues.**\n\n"
    else:
        documentation += "❌ **Both phases encountered issues.**\n\n"
    
    # Critical issues
    documentation += "### Critical Issues\n\n"
    
    critical_issues = []
    
    if phase2_results:
        phase2_tests = phase2_results.get('tests', [])
        for test in phase2_tests:
            if test.get('status') == 'fail':
                critical_issues.append({
                    'phase': 'Phase 2',
                    'component': test.get('name', 'Unknown'),
                    'error': test.get('error', 'No error message')
                })
    
    if phase3_results:
        phase3_results_dict = phase3_results.get('test_results', phase3_results.get('results', {}))
        phase3_details = phase3_results.get('test_details', {})
        for component_key, passed in phase3_results_dict.items():
            if not passed:
                component_name = {
                    'memory_cache': 'Memory Cache Logic',
                    'orchestrator': 'Orchestrator Logic',
                    'bandit': 'Bandit Logic',
                    'integration': 'Integration Logic',
                    'token_counting': 'Token Counting',
                    'bandit_algorithms': 'Bandit Algorithms'
                }.get(component_key, component_key)
                error_msg = "Component validation failed"
                if component_key in phase3_details and 'error' in phase3_details[component_key]:
                    error_msg = phase3_details[component_key]['error']
                critical_issues.append({
                    'phase': 'Phase 3',
                    'component': component_name,
                    'error': error_msg
                })
    
    if critical_issues:
        for issue in critical_issues:
            documentation += f"- **{issue['phase']} - {issue['component']}**: {issue['error']}\n"
        documentation += "\n"
    else:
        documentation += "No critical issues found.\n\n"
    
    # Recommendations
    documentation += "## Unified Recommendations\n\n"
    
    if phase2_summary.get('pass_rate', 0) < 100 or phase3_passed < phase3_total:
        documentation += "### Immediate Actions\n\n"
        
        if phase2_summary.get('failed', 0) > 0:
            documentation += "1. **Address Phase 2 Failures**: Review failed integration tests and fix underlying issues.\n"
        
        if phase3_passed < phase3_total:
            documentation += "2. **Fix Component Logic**: Address failed component validation tests.\n"
        
        documentation += "\n"
    
    if phase2_summary.get('warnings', 0) > 0:
        documentation += "### Optimization Opportunities\n\n"
        documentation += "Phase 2 tests completed with warnings. Review these for potential improvements:\n\n"
        if phase2_results:
            for test in phase2_results.get('tests', []):
                if test.get('status') == 'warning':
                    documentation += f"- {test.get('name', 'Unknown')}\n"
        documentation += "\n"
    
    if phase2_ok and phase3_ok:
        documentation += "✅ **All validation tests passed!** The platform is ready for production use.\n\n"
    
    # Next steps
    documentation += "## Next Steps\n\n"
    documentation += "1. Review detailed phase documentation:\n"
    documentation += "   - [Phase 2 Results](PHASE2_RESULTS.md)\n"
    documentation += "   - [Phase 3 Results](PHASE3_RESULTS.md)\n"
    documentation += "2. Address any critical issues identified above\n"
    documentation += "3. Generate unified validation report:\n"
    documentation += "   ```bash\n"
    documentation += "   python3 scripts/generate_validation_report.py\n"
    documentation += "   ```\n"
    
    # Save documentation
    doc_file.parent.mkdir(exist_ok=True)
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write(documentation)
    
    print(f"\n✓ Combined summary documented in: {doc_file}")
    return doc_file


def main():
    """Main execution."""
    print("=" * 80)
    print("EXECUTING PHASES 2 AND 3: PLATFORM VALIDATION")
    print("=" * 80)
    print()
    
    # Check prerequisites
    check_prerequisites()
    
    # Execute Phase 2
    phase2_results, phase2_success = run_phase2()
    
    # Document Phase 2
    phase2_doc = document_phase2(phase2_results, phase2_success)
    
    # Execute Phase 3
    phase3_results, phase3_success = run_phase3()
    
    # Document Phase 3
    phase3_doc = document_phase3(phase3_results, phase3_success)
    
    # Create combined summary
    combined_doc = create_combined_summary(phase2_results, phase2_success, phase3_results, phase3_success)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)
    print()
    
    print("Phase 2 Results:")
    if phase2_results:
        summary = phase2_results.get('summary', {})
        print(f"  Status: {'✅ PASSED' if phase2_success and summary.get('failed', 0) == 0 else '⚠️ ISSUES' if phase2_success else '❌ FAILED'}")
        print(f"  Tests: {summary.get('passed', 0)}/{summary.get('total_tests', 0)} passed")
        print(f"  Documentation: {phase2_doc}")
    else:
        print("  Status: ⚠️ No results available")
    
    print()
    print("Phase 3 Results:")
    if phase3_results:
        phase3_results_dict = phase3_results.get('test_results', phase3_results.get('results', {}))
        phase3_passed = sum(1 for v in phase3_results_dict.values() if v)
        phase3_total = len(phase3_results_dict)
        print(f"  Status: {'✅ PASSED' if phase3_success and phase3_passed == phase3_total else '⚠️ ISSUES' if phase3_success else '❌ FAILED'}")
        print(f"  Tests: {phase3_passed}/{phase3_total} passed")
        print(f"  Documentation: {phase3_doc}")
    else:
        print("  Status: ⚠️ No results available")
    
    print()
    print(f"Combined Summary: {combined_doc}")
    print()
    print("Next Step: Generate unified validation report")
    print("  python3 scripts/generate_validation_report.py")
    print("=" * 80)
    
    # Return success if both phases completed (even if with issues)
    return 0 if (phase2_success or phase2_results is not None) and (phase3_success or phase3_results is not None) else 1


if __name__ == "__main__":
    sys.exit(main())

