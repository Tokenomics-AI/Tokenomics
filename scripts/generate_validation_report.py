#!/usr/bin/env python3
"""Generate unified validation report from all test suite results."""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ValidationReportGenerator:
    """Generates unified validation report from all test results."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "validation_results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.ml_evaluation_results = None
        self.platform_integration_results = None
        self.component_validation_results = None
    
    def load_ml_evaluation_results(self) -> bool:
        """Load ML evaluation results."""
        results_file = self.project_root / "training_data" / "evaluation_report.json"
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    self.ml_evaluation_results = json.load(f)
                return True
            except Exception as e:
                print(f"⚠ Warning: Failed to load ML evaluation results: {e}")
                return False
        else:
            print(f"⚠ Warning: ML evaluation results not found: {results_file}")
            return False
    
    def load_platform_integration_results(self) -> bool:
        """Load platform integration test results."""
        results_file = self.project_root / "tests" / "integration" / "comprehensive_test_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    self.platform_integration_results = json.load(f)
                return True
            except Exception as e:
                print(f"⚠ Warning: Failed to load platform integration results: {e}")
                return False
        else:
            print(f"⚠ Warning: Platform integration results not found: {results_file}")
            return False
    
    def load_component_validation_results(self) -> bool:
        """Load component validation results."""
        results_file = self.project_root / "training_data" / "component_validation_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    self.component_validation_results = json.load(f)
                return True
            except Exception as e:
                print(f"⚠ Warning: Failed to load component validation results: {e}")
                return False
        else:
            print(f"⚠ Warning: Component validation results not found: {results_file}")
            return False
    
    def calculate_platform_health_score(self) -> float:
        """Calculate overall platform health score (0-100)."""
        scores = []
        
        # ML Models score (0-40 points)
        if self.ml_evaluation_results:
            ml_score = 0
            if 'token_predictor' in self.ml_evaluation_results:
                token_metrics = self.ml_evaluation_results['token_predictor']
                if 'error' not in token_metrics:
                    # Score based on accuracy within 20%
                    accuracy = token_metrics.get('accuracy_within_20pct', 0)
                    ml_score += (accuracy / 100) * 15  # 15 points max
            
            if 'complexity_classifier' in self.ml_evaluation_results:
                complexity_metrics = self.ml_evaluation_results['complexity_classifier']
                if 'error' not in complexity_metrics:
                    accuracy = complexity_metrics.get('accuracy', 0)
                    ml_score += (accuracy / 100) * 15  # 15 points max
            
            if 'escalation_predictor' in self.ml_evaluation_results:
                escalation_metrics = self.ml_evaluation_results['escalation_predictor']
                if 'error' not in escalation_metrics and 'accuracy' in escalation_metrics:
                    accuracy = escalation_metrics.get('accuracy', 0)
                    ml_score += (accuracy / 100) * 10  # 10 points max
            
            scores.append(('ML Models', ml_score, 40))
        
        # Platform Integration score (0-40 points)
        if self.platform_integration_results:
            platform_score = 0
            summary = self.platform_integration_results.get('summary', {})
            total_tests = summary.get('total', 0)
            passed_tests = summary.get('passed', 0)
            if total_tests > 0:
                platform_score = (passed_tests / total_tests) * 40
            scores.append(('Platform Integration', platform_score, 40))
        
        # Component Validation score (0-20 points)
        if self.component_validation_results:
            component_score = 0
            summary = self.component_validation_results.get('summary', {})
            total_tests = summary.get('total_tests', 0)
            passed_tests = summary.get('passed', 0)
            if total_tests > 0:
                component_score = (passed_tests / total_tests) * 20
            scores.append(('Component Validation', component_score, 20))
        
        total_score = sum(score for _, score, _ in scores)
        max_possible = sum(max_points for _, _, max_points in scores)
        
        if max_possible > 0:
            health_score = (total_score / max_possible) * 100
        else:
            health_score = 0.0
        
        return round(health_score, 2)
    
    def generate_json_report(self) -> Dict[str, Any]:
        """Generate unified JSON report."""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'platform_health_score': self.calculate_platform_health_score(),
            'ml_evaluation': self.ml_evaluation_results,
            'platform_integration': self.platform_integration_results,
            'component_validation': self.component_validation_results,
            'summary': {
                'ml_evaluation_available': self.ml_evaluation_results is not None,
                'platform_integration_available': self.platform_integration_results is not None,
                'component_validation_available': self.component_validation_results is not None,
            }
        }
        
        return report
    
    def generate_markdown_report(self, json_report: Dict[str, Any]) -> str:
        """Generate unified Markdown report."""
        md = []
        md.append("# Complete Platform Validation Report\n")
        md.append(f"**Generated**: {json_report['report_timestamp']}\n")
        md.append(f"**Platform Health Score**: {json_report['platform_health_score']}%\n")
        md.append("\n---\n")
        
        # Executive Summary
        md.append("## Executive Summary\n")
        md.append(f"- **Overall Status**: {'✅ PASS' if json_report['platform_health_score'] >= 70 else '⚠️ WARNING' if json_report['platform_health_score'] >= 50 else '❌ FAIL'}\n")
        md.append(f"- **Platform Health Score**: {json_report['platform_health_score']}%\n")
        
        available_tests = []
        if json_report['summary']['ml_evaluation_available']:
            available_tests.append('ML Evaluation')
        if json_report['summary']['platform_integration_available']:
            available_tests.append('Platform Integration')
        if json_report['summary']['component_validation_available']:
            available_tests.append('Component Validation')
        
        md.append(f"- **Test Coverage**: {len(available_tests)}/3 test suites completed\n")
        md.append(f"  - Available: {', '.join(available_tests)}\n")
        md.append("\n---\n")
        
        # ML Models Performance
        if self.ml_evaluation_results:
            md.append("## ML Models Performance\n")
            
            # Token Predictor
            if 'token_predictor' in self.ml_evaluation_results:
                token_metrics = self.ml_evaluation_results['token_predictor']
                if 'error' not in token_metrics:
                    md.append("### Token Predictor\n")
                    md.append(f"- **MAE**: {token_metrics.get('mae', 0):.2f} tokens\n")
                    md.append(f"- **RMSE**: {token_metrics.get('rmse', 0):.2f} tokens\n")
                    md.append(f"- **MAPE**: {token_metrics.get('mape', 0):.2f}%\n")
                    md.append(f"- **Accuracy (within 20%)**: {token_metrics.get('accuracy_within_20pct', 0):.2f}%\n")
                    md.append(f"- **Samples Evaluated**: {token_metrics.get('num_samples', 0)}\n")
                    md.append("\n")
            
            # Complexity Classifier
            if 'complexity_classifier' in self.ml_evaluation_results:
                complexity_metrics = self.ml_evaluation_results['complexity_classifier']
                if 'error' not in complexity_metrics:
                    md.append("### Complexity Classifier\n")
                    md.append(f"- **Overall Accuracy**: {complexity_metrics.get('accuracy', 0):.2f}%\n")
                    if 'per_class_accuracy' in complexity_metrics:
                        md.append("- **Per-Class Accuracy**:\n")
                        for cls, acc in complexity_metrics['per_class_accuracy'].items():
                            md.append(f"  - {cls.capitalize()}: {acc:.2f}%\n")
                    md.append(f"- **Samples Evaluated**: {complexity_metrics.get('num_samples', 0)}\n")
                    md.append("\n")
            
            # Escalation Predictor
            if 'escalation_predictor' in self.ml_evaluation_results:
                escalation_metrics = self.ml_evaluation_results['escalation_predictor']
                md.append("### Escalation Predictor\n")
                if 'error' not in escalation_metrics and 'accuracy' in escalation_metrics:
                    md.append(f"- **Accuracy**: {escalation_metrics.get('accuracy', 0):.2f}%\n")
                    md.append(f"- **Precision**: {escalation_metrics.get('precision', 0):.2f}%\n")
                    md.append(f"- **Recall**: {escalation_metrics.get('recall', 0):.2f}%\n")
                    md.append(f"- **F1 Score**: {escalation_metrics.get('f1_score', 0):.2f}%\n")
                else:
                    md.append("- Evaluation requires cascading outcome analysis\n")
                md.append("\n")
            
            md.append("---\n")
        
        # Platform Components Performance
        if self.platform_integration_results:
            md.append("## Platform Components Performance\n")
            
            tests = self.platform_integration_results.get('tests', [])
            for test in tests:
                test_name = test.get('name', 'Unknown')
                status = test.get('status', 'unknown')
                status_icon = '✅' if status == 'pass' else '❌' if status == 'fail' else '⚠️'
                md.append(f"### {test_name}\n")
                md.append(f"- **Status**: {status_icon} {status.upper()}\n")
                
                details = test.get('details', {})
                if details:
                    for key, value in details.items():
                        md.append(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                md.append("\n")
            
            md.append("---\n")
        
        # Component Validation
        if self.component_validation_results:
            md.append("## Component Validation\n")
            
            summary = self.component_validation_results.get('summary', {})
            md.append(f"- **Tests Passed**: {summary.get('passed', 0)}/{summary.get('total_tests', 0)}\n")
            md.append("\n")
            
            test_results = self.component_validation_results.get('test_results', {})
            for test_name, result in test_results.items():
                status_icon = '✅' if result else '❌'
                md.append(f"- **{test_name.replace('_', ' ').title()}**: {status_icon} {'PASS' if result else 'FAIL'}\n")
            
            md.append("\n---\n")
        
        # Recommendations
        md.append("## Recommendations\n")
        
        health_score = json_report['platform_health_score']
        if health_score >= 90:
            md.append("- ✅ Platform is in excellent condition. Continue monitoring.\n")
        elif health_score >= 70:
            md.append("- ⚠️ Platform is functioning well but has room for improvement.\n")
            md.append("- Consider collecting more training data for ML models.\n")
        elif health_score >= 50:
            md.append("- ⚠️ Platform needs attention. Review failing components.\n")
            md.append("- Prioritize fixing failing tests.\n")
            md.append("- Consider retraining ML models with more data.\n")
        else:
            md.append("- ❌ Platform requires immediate attention.\n")
            md.append("- Review and fix all failing components.\n")
            md.append("- Re-run validation after fixes.\n")
        
        md.append("\n---\n")
        md.append(f"*Report generated at {json_report['report_timestamp']}*\n")
        
        return ''.join(md)
    
    def save_reports(self):
        """Save JSON and Markdown reports."""
        # Generate JSON report
        json_report = self.generate_json_report()
        
        json_file = self.results_dir / "unified_validation_report.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, default=str)
        print(f"✓ JSON report saved: {json_file}")
        
        # Generate Markdown report
        md_report = self.generate_markdown_report(json_report)
        
        md_file = self.results_dir / "unified_validation_report.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)
        print(f"✓ Markdown report saved: {md_file}")
        
        return json_file, md_file
    
    def generate(self):
        """Generate unified validation report."""
        print("=" * 80)
        print("GENERATING UNIFIED VALIDATION REPORT")
        print("=" * 80)
        print()
        
        # Load all results
        print("Loading test results...")
        ml_loaded = self.load_ml_evaluation_results()
        platform_loaded = self.load_platform_integration_results()
        component_loaded = self.load_component_validation_results()
        
        print()
        print(f"ML Evaluation: {'✓ Loaded' if ml_loaded else '✗ Not available'}")
        print(f"Platform Integration: {'✓ Loaded' if platform_loaded else '✗ Not available'}")
        print(f"Component Validation: {'✓ Loaded' if component_loaded else '✗ Not available'}")
        print()
        
        # Generate and save reports
        json_file, md_file = self.save_reports()
        
        # Calculate health score
        health_score = self.calculate_platform_health_score()
        
        print()
        print("=" * 80)
        print("REPORT GENERATION COMPLETE")
        print("=" * 80)
        print(f"Platform Health Score: {health_score}%")
        print(f"JSON Report: {json_file}")
        print(f"Markdown Report: {md_file}")
        print("=" * 80)
        
        return json_file, md_file


if __name__ == "__main__":
    generator = ValidationReportGenerator()
    generator.generate()
