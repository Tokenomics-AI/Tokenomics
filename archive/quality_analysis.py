"""Quality analysis tool for comparing response quality."""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def analyze_quality(report_file: str) -> Dict:
    """Analyze quality metrics from usage report."""
    with open(report_file, 'r') as f:
        data = json.load(f)
    
    records = data['records']
    summary = data['summary']
    
    # Group by cache status
    cached_responses = [r for r in records if r['cache_hit']]
    non_cached_responses = [r for r in records if not r['cache_hit']]
    
    # Quality metrics
    analysis = {
        'total_responses': len(records),
        'cached_count': len(cached_responses),
        'non_cached_count': len(non_cached_responses),
        'average_quality': summary.get('average_quality_score', 0),
        'cached_quality': summary.get('cached_quality_score', 0),
        'non_cached_quality': summary.get('non_cached_quality_score', 0),
        'quality_preservation': 0.0,
    }
    
    if cached_responses and non_cached_responses:
        # Check if cached responses maintain quality
        analysis['quality_preservation'] = (
            analysis['cached_quality'] / analysis['non_cached_quality']
            if analysis['non_cached_quality'] > 0 else 1.0
        )
    
    # Detailed metrics
    analysis['detailed_metrics'] = {
        'cached': {
            'avg_length': sum(r['response_length'] for r in cached_responses) / len(cached_responses) if cached_responses else 0,
            'avg_words': sum(r.get('quality_metrics', {}).get('word_count', 0) for r in cached_responses) / len(cached_responses) if cached_responses else 0,
            'avg_sentences': sum(r.get('quality_metrics', {}).get('sentence_count', 0) for r in cached_responses) / len(cached_responses) if cached_responses else 0,
        },
        'non_cached': {
            'avg_length': sum(r['response_length'] for r in non_cached_responses) / len(non_cached_responses) if non_cached_responses else 0,
            'avg_words': sum(r.get('quality_metrics', {}).get('word_count', 0) for r in non_cached_responses) / len(non_cached_responses) if non_cached_responses else 0,
            'avg_sentences': sum(r.get('quality_metrics', {}).get('sentence_count', 0) for r in non_cached_responses) / len(non_cached_responses) if non_cached_responses else 0,
        },
    }
    
    return analysis


def compare_reports(report_with_cache: str, report_without_cache: str) -> Dict:
    """Compare quality between cached and non-cached runs."""
    with_cache = analyze_quality(report_with_cache)
    without_cache = analyze_quality(report_without_cache)
    
    comparison = {
        'with_cache': with_cache,
        'without_cache': without_cache,
        'quality_comparison': {
            'cached_vs_fresh': with_cache['cached_quality'] - without_cache['non_cached_quality'],
            'preservation_rate': with_cache['quality_preservation'],
        }
    }
    
    return comparison


def print_quality_report(analysis: Dict):
    """Print formatted quality report."""
    print("\n" + "=" * 80)
    print("QUALITY ANALYSIS REPORT")
    print("=" * 80)
    print()
    print("OVERALL QUALITY:")
    print(f"  Average Quality Score: {analysis['average_quality']:.3f}/1.0")
    print(f"  Cached Responses: {analysis['cached_quality']:.3f}/1.0")
    print(f"  Non-Cached Responses: {analysis['non_cached_quality']:.3f}/1.0")
    print(f"  Quality Preservation: {analysis['quality_preservation']*100:.1f}%")
    print()
    print("DETAILED METRICS:")
    print("  Cached Responses:")
    cached_metrics = analysis['detailed_metrics']['cached']
    print(f"    Average Length: {cached_metrics['avg_length']:.0f} characters")
    print(f"    Average Words: {cached_metrics['avg_words']:.0f} words")
    print(f"    Average Sentences: {cached_metrics['avg_sentences']:.1f} sentences")
    print("  Non-Cached Responses:")
    non_cached_metrics = analysis['detailed_metrics']['non_cached']
    print(f"    Average Length: {non_cached_metrics['avg_length']:.0f} characters")
    print(f"    Average Words: {non_cached_metrics['avg_words']:.0f} words")
    print(f"    Average Sentences: {non_cached_metrics['avg_sentences']:.1f} sentences")
    print("=" * 80)


if __name__ == "__main__":
    # Analyze cached report
    print("Analyzing quality metrics...")
    
    if Path("usage_report_with_cache.json").exists():
        with_cache_analysis = analyze_quality("usage_report_with_cache.json")
        print_quality_report(with_cache_analysis)
    
    # Compare if both reports exist
    if Path("usage_report_with_cache.json").exists() and Path("usage_report_without_cache.json").exists():
        comparison = compare_reports("usage_report_with_cache.json", "usage_report_without_cache.json")
        print("\n" + "=" * 80)
        print("QUALITY COMPARISON")
        print("=" * 80)
        print(f"Cached vs Fresh Quality Difference: {comparison['quality_comparison']['cached_vs_fresh']:+.3f}")
        print(f"Quality Preservation Rate: {comparison['quality_comparison']['preservation_rate']*100:.1f}%")
        print("=" * 80)

