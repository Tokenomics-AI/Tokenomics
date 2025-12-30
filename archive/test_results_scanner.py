"""Scan and parse test results from various locations."""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


def parse_markdown_metrics(content: str) -> Dict:
    """Parse metrics from markdown content."""
    metrics = {}
    
    # Extract key-value pairs from markdown
    patterns = {
        'total_queries': r'\*\*Total Queries:\*\*\s*(\d+)',
        'successful_queries': r'\*\*Successful Queries:\*\*\s*(\d+)',
        'failed_queries': r'\*\*Failed Queries:\*\*\s*(\d+)',
        'token_savings': r'\*\*Token Savings:\*\*\s*([\d,]+)\s*tokens\s*\(([\d.]+)%\)',
        'cost_savings': r'\*\*Cost Savings:\*\*\s*\$\s*([\d.]+)\s*\(([\d.]+)%\)',
        'cache_hit_rate': r'\*\*Cache Hit Rate:\*\*\s*([\d.]+)%',
        'quality_preservation': r'\*\*Quality Preservation Rate:\*\*\s*([\d.]+)%',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            if key in ['token_savings', 'cost_savings']:
                metrics[key] = {
                    'value': match.group(1).replace(',', ''),
                    'percent': match.group(2)
                }
            else:
                metrics[key] = match.group(1)
    
    return metrics


def scan_test_results() -> List[Dict]:
    """Scan all test result files and return structured data."""
    project_root = Path(__file__).parent
    results = []
    
    # Define test result locations
    test_locations = [
        {
            'name': 'A/B Comparison Test',
            'type': 'ab_comparison',
            'paths': [
                (project_root / 'tests' / 'results', 'AB_COMPARISON_REPORT_*.md'),
                (project_root / 'tests' / 'results', 'ab_comparison_results_*.json'),
            ],
            'description': 'Comprehensive A/B test comparing baseline vs optimized queries across 50+ diverse prompts',
        },
        {
            'name': 'Platform Validation Suite',
            'type': 'validation',
            'paths': [
                (project_root, 'VALIDATION_SUITE_RESULTS.md'),
            ],
            'description': '60 controlled prompts to verify Memory, Routing, and Compression components',
        },
        {
            'name': 'Comprehensive Diagnostic Test',
            'type': 'diagnostic',
            'paths': [
                (project_root / 'tests' / 'diagnostic' / 'results', 'extensive_diagnostic_*.md'),
                (project_root / 'tests' / 'diagnostic' / 'results', 'extensive_diagnostic_*.json'),
            ],
            'description': 'Extensive diagnostic test to trigger all platform components',
        },
        {
            'name': 'Performance Impact Analysis',
            'type': 'analysis',
            'paths': [
                (project_root / 'tests' / 'diagnostic' / 'results', 'PERFORMANCE_IMPACT_ANALYSIS.md'),
            ],
            'description': 'Analysis of performance improvements after fixes',
        },
        {
            'name': 'Support Benchmark',
            'type': 'benchmark',
            'paths': [
                (project_root / 'tests' / 'benchmarks' / 'results', 'support_benchmark_results.json'),
            ],
            'description': 'Benchmark test using support dataset',
        },
        {
            'name': 'Quick Benchmark',
            'type': 'benchmark',
            'paths': [
                (project_root / 'tests' / 'benchmarks' / 'results', 'quick_benchmark_results.json'),
            ],
            'description': 'Quick benchmark test for rapid validation',
        },
    ]
    
    for test_info in test_locations:
        # Find all matching files first
        all_files = []
        for parent_dir, pattern in test_info['paths']:
            if '*' in pattern:
                # Glob pattern
                files = list(parent_dir.glob(pattern))
            else:
                # Direct path
                file_path = parent_dir / pattern
                files = [file_path] if file_path.exists() else []
            all_files.extend(files)
        
        # Group files by date to create separate test results
        files_by_date = {}
        for file_path in all_files:
            if not file_path.exists():
                continue
                
            # Extract date from filename or content
            date = None
            if file_path.suffix == '.json':
                try:
                    data = json.loads(file_path.read_text())
                    if 'metadata' in data and 'generated_at' in data['metadata']:
                        date = data['metadata']['generated_at'][:10]
                except:
                    pass
            
            if not date:
                # Try to extract from filename
                date_match = re.search(r'(\d{8})', file_path.name)
                if date_match:
                    date_str = date_match.group(1)
                    date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            date_key = date or 'unknown'
            if date_key not in files_by_date:
                files_by_date[date_key] = []
            files_by_date[date_key].append(file_path)
        
        # Create test result for each date group
        for date_key, files in files_by_date.items():
            test_result = {
                'id': f"{test_info['type']}_{date_key}",
                'name': test_info['name'],
                'type': test_info['type'],
                'description': test_info['description'],
                'files': [],
                'metrics': {},
                'content': None,
                'date': date_key if date_key != 'unknown' else None,
            }
            
            for file_path in files:
                test_result['files'].append({
                    'path': str(file_path),
                    'name': file_path.name,
                    'type': 'markdown' if file_path.suffix == '.md' else 'json',
                })
                
                # Parse file
                try:
                    if file_path.suffix == '.md':
                        content = file_path.read_text()
                        if not test_result['content']:
                            test_result['content'] = content
                        test_result['metrics'].update(parse_markdown_metrics(content))
                        
                        # Extract date from content if not already set
                        if not test_result['date']:
                            date_match = re.search(r'\*\*Generated:\*\*\s*(\d{4}-\d{2}-\d{2})', content)
                            if date_match:
                                test_result['date'] = date_match.group(1)
                    
                    elif file_path.suffix == '.json':
                        try:
                            data = json.loads(file_path.read_text())
                            test_result['json_data'] = data
                            
                            # Extract metrics from JSON
                            if 'summary' in data:
                                test_result['metrics'].update(data['summary'])
                            if 'aggregated_metrics' in data:
                                test_result['metrics'].update(data['aggregated_metrics'])
                            if 'metadata' in data:
                                if 'generated_at' in data['metadata']:
                                    test_result['date'] = data['metadata']['generated_at'][:10]
                        except json.JSONDecodeError as e:
                            # Skip invalid JSON files
                            print(f"Warning: Invalid JSON in {file_path}: {e}")
                            continue
                
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
                    continue
            
            if test_result['files']:
                results.append(test_result)
    
    # Sort by date (most recent first), handling None dates
    results.sort(key=lambda x: x.get('date') or '', reverse=True)
    
    return results


def get_test_details(test_id: str) -> Optional[Dict]:
    """Get detailed information for a specific test."""
    all_tests = scan_test_results()
    
    for test in all_tests:
        if test['id'] == test_id:
            # Load full content if not already loaded
            if test.get('content') is None and test.get('files'):
                for file_info in test['files']:
                    if file_info['type'] == 'markdown':
                        try:
                            file_path = Path(file_info['path'])
                            if file_path.exists():
                                test['content'] = file_path.read_text()
                                break
                        except:
                            pass
            
            return test
    
    return None







