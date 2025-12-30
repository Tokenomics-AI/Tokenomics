"""Flask backend API for Tokenomics frontend."""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
from flask import Flask, render_template, request, jsonify, Response, redirect
from flask_cors import CORS
import threading
import structlog

sys.path.insert(0, str(Path(__file__).parent))

from tokenomics.core import TokenomicsPlatform
from tokenomics.config import TokenomicsConfig
from usage_tracker import UsageTracker
from test_results_scanner import scan_test_results, get_test_details

logger = structlog.get_logger()

app = Flask(__name__)
CORS(app)

# Global state
platform = None
current_run = None
runs_history = []

# Metrics tracking
metrics_history = {
    "total_queries": 0,
    "total_cost_savings": 0.0,
    "total_baseline_cost": 0.0,
    "total_optimized_cost": 0.0,
    "cache_hits": 0,
    "cache_misses": 0,
    "quality_scores": [],
    "component_savings": {
        "memory_layer": 0.0,
        "bandit": 0.0,
        "orchestrator": 0.0,
    },
    "strategy_distribution": {},
    "model_distribution": {},
}


def check_api_key():
    """Check if API key is configured properly."""
    env_file = Path(__file__).parent / '.env'
    
    # Check if .env file exists
    if not env_file.exists():
        return False, f".env file not found at {env_file}. Please create it from env.template."
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    
    if not api_key:
        return False, "OPENAI_API_KEY not found in environment variables. Please set it in your .env file."
    
    if len(api_key) < 10:  # Basic validation
        return False, "OPENAI_API_KEY appears to be invalid (too short). Please check your .env file."
    
    return True, "API key configured"


def get_env_status() -> dict:
    """Get status of environment configuration."""
    env_file = Path(__file__).parent / '.env'
    env_exists = env_file.exists()
    
    status = {
        'env_file_exists': env_exists,
        'env_file_path': str(env_file),
        'api_key_configured': False,
        'error': None
    }
    
    if env_exists:
        from dotenv import load_dotenv
        load_dotenv(env_file)
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        status['api_key_configured'] = bool(api_key and len(api_key) >= 10)
        if not status['api_key_configured']:
            status['error'] = "OPENAI_API_KEY not set or invalid in .env file"
    else:
        status['error'] = f".env file not found at {env_file}"
    
    return status


def init_platform():
    """Initialize Tokenomics platform."""
    global platform
    
    if platform is None:
        # Check API key first
        is_valid, error_msg = check_api_key()
        if not is_valid:
            raise ValueError(error_msg)
        
        config = TokenomicsConfig.from_env()
        # Override to use OpenAI
        config.llm.provider = "openai"
        config.llm.model = "gpt-4o-mini"  # Using a cost-effective model
        config.llm.api_key = os.environ["OPENAI_API_KEY"].strip()
        
        # Enable semantic cache for similar query matching
        config.memory.use_semantic_cache = True
        config.memory.cache_size = 100
        # Lower thresholds to catch more semantic matches - these queries should match!
        config.memory.similarity_threshold = 0.65  # Context threshold (lowered to catch more matches)
        config.memory.direct_return_threshold = 0.75  # Direct return threshold (lowered to catch more matches)
        
        platform = TokenomicsPlatform(config=config)
    
    return platform


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Redirect dashboard to playground."""
    return redirect('/playground')


@app.route('/playground')
def playground():
    """Tokenomics Playground page."""
    return render_template('playground.html')


@app.route('/results')
def results():
    """Test results page."""
    return render_template('results.html')




@app.route('/api/run', methods=['POST'])
def run_experiment():
    """Run an experiment."""
    global current_run, platform
    
    try:
        data = request.json
        queries = data.get('queries', [])
        mode = data.get('mode', 'tokenomics')  # baseline, tokenomics, ab
        num_queries = data.get('num_queries', 1)
        
        if not queries:
            return jsonify({'error': 'No queries provided'}), 400
        
        # Validate queries list
        if isinstance(queries, list):
            queries = [q.strip() for q in queries if q and q.strip()]
            if not queries:
                return jsonify({'error': 'No valid queries provided'}), 400
        else:
            queries = [str(queries).strip()]
            if not queries[0]:
                return jsonify({'error': 'Query cannot be empty'}), 400
        
        # Initialize platform
        try:
            platform = init_platform()
        except ValueError as e:
            return jsonify({
                'error': 'API key not configured',
                'error_type': 'missing_api_key',
                'message': str(e),
                'env_status': get_env_status(),
            }), 400
        
        # Create run
        run_id = f"run_{int(time.time())}"
        current_run = {
            'id': run_id,
            'mode': mode,
            'queries': queries[:num_queries] if isinstance(queries, list) else [queries] * num_queries,
            'start_time': datetime.now().isoformat(),
            'results': [],
            'summary': {},
        }
        
        # Run queries
        # IMPORTANT: Platform instance is shared across all queries in a run
        # This means cache accumulates as we process queries, allowing later queries
        # to benefit from cache hits on earlier queries
        tracker = UsageTracker()
        total_tokens = 0
        total_latency = 0
        cache_hits = 0
        
        for i, query in enumerate(current_run['queries'], 1):
            start_time = time.time()
            
            try:
                if mode == 'baseline':
                    # Baseline mode - no optimization
                    result = platform.query(query, use_cache=False, use_bandit=False)
                elif mode == 'ab':
                    # A/B Comparison - use compare_with_baseline() method
                    # This explicitly runs both baseline and optimized paths
                    optimized_result = platform.compare_with_baseline(query)
                    
                    # Extract baseline data from comparison result
                    baseline_comparison = optimized_result.get("baseline_comparison_result", {})
                    if baseline_comparison:
                        baseline_result = {
                            'tokens_used': baseline_comparison.get("tokens_used", 0),
                            'input_tokens': baseline_comparison.get("input_tokens", 0),
                            'output_tokens': baseline_comparison.get("output_tokens", 0),
                            'latency_ms': baseline_comparison.get("latency_ms", 0),
                            'model': baseline_comparison.get("model", "unknown"),
                            'response': baseline_comparison.get("response", ""),
                        }
                    else:
                        # Fallback: run baseline separately if comparison data not available
                        baseline_result = platform._run_baseline_query(query)
                    
                    # Log cache state for debugging
                    if i > 1:  # For queries after the first one
                        cache_stats = platform.memory.stats()
                        logger.info(
                            "Cache state before query",
                            query_num=i,
                            cache_size=cache_stats.get('size', 0),
                            cache_max_size=cache_stats.get('max_size', 0),
                        )
                    
                    # Extract optimized metrics
                    optimized_tokens = optimized_result['tokens_used']
                    optimized_input = optimized_result.get('input_tokens', 0)
                    optimized_output = optimized_result.get('output_tokens', 0)
                    optimized_latency = optimized_result['latency_ms']
                    optimized_component_savings = optimized_result.get('component_savings', {})
                    
                    baseline_tokens = baseline_result['tokens_used']
                    
                    # Calculate savings
                    token_savings = baseline_tokens - optimized_tokens
                    token_savings_pct = (token_savings / baseline_tokens * 100) if baseline_tokens > 0 else 0
                    latency_reduction = baseline_result['latency_ms'] - optimized_latency
                    latency_reduction_pct = (latency_reduction / baseline_result['latency_ms'] * 100) if baseline_result['latency_ms'] > 0 else 0
                    
                    # Serialize plans for both baseline and optimized
                    baseline_plan_data = None
                    if baseline_result.get('plan'):
                        bp = baseline_result['plan']
                        baseline_plan_data = {
                            'complexity': bp.complexity.value if hasattr(bp.complexity, 'value') else str(bp.complexity),
                            'token_budget': bp.token_budget,
                            'model': bp.model,
                            'allocations': [
                                {
                                    'component': alloc.component,
                                    'tokens': alloc.tokens,
                                    'priority': alloc.priority,
                                    'utility': alloc.utility,
                                }
                                for alloc in bp.allocations
                            ],
                        }
                    
                    optimized_plan_data = None
                    if optimized_result.get('plan'):
                        op = optimized_result['plan']
                        optimized_plan_data = {
                            'complexity': op.complexity.value if hasattr(op.complexity, 'value') else str(op.complexity),
                            'token_budget': op.token_budget,
                            'model': op.model,
                            'allocations': [
                                {
                                    'component': alloc.component,
                                    'tokens': alloc.tokens,
                                    'priority': alloc.priority,
                                    'utility': alloc.utility,
                                }
                                for alloc in op.allocations
                            ],
                        }
                    
                    # Shorten query text for diagnostics (first 100 chars or full if < 100)
                    query_text_short = query[:100] if len(query) > 100 else query
                    
                    result = {
                        'baseline': {
                            'tokens_used': baseline_result['tokens_used'],
                            'latency_ms': baseline_result['latency_ms'],
                            'response': baseline_result.get('response', ''),
                            'response_length': len(baseline_result.get('response', '')),
                            'plan': baseline_plan_data,
                        },
                        'optimized': {
                            'tokens_used': optimized_tokens,
                            'input_tokens': optimized_input,
                            'output_tokens': optimized_output,
                            'latency_ms': optimized_latency,
                            'cache_hit': optimized_result.get('cache_hit', False),
                            'cache_type': optimized_result.get('cache_type', 'none'),
                            'similarity': optimized_result.get('similarity'),
                            'strategy': optimized_result.get('strategy', 'none'),
                            'model': optimized_result.get('model', platform.config.llm.model),
                            'response': optimized_result.get('response', ''),
                            'response_length': len(optimized_result.get('response', '')),
                            'plan': optimized_plan_data,
                            'reward': optimized_result.get('reward'),
                            'component_savings': optimized_component_savings,
                            'quality_judge': optimized_result.get('quality_judge'),
                            # Diagnostic fields
                            'query_text': query_text_short,
                            'query_type': optimized_result.get('query_type'),
                            'cache_tier': optimized_result.get('cache_tier', 'none'),
                            'capsule_tokens': optimized_result.get('capsule_tokens', 0),
                            'strategy_arm': optimized_result.get('strategy_arm'),
                            'model_used': optimized_result.get('model_used') or optimized_result.get('model', platform.config.llm.model),
                            'used_memory': optimized_result.get('used_memory', False),
                            'user_preference': optimized_result.get('user_preference'),
                        },
                        'improvement': {
                            'token_savings': token_savings,
                            'token_savings_percent': round(token_savings_pct, 2),
                            'latency_reduction': round(latency_reduction, 2),
                            'latency_reduction_percent': round(latency_reduction_pct, 2),
                        },
                        'tokens_used': optimized_result['tokens_used'],  # Use optimized for summary
                        'latency_ms': optimized_result['latency_ms'],
                        'cache_hit': optimized_result['cache_hit'],
                        'cache_type': optimized_result.get('cache_type', 'none'),
                        'strategy': optimized_result.get('strategy', 'none'),
                        'model': optimized_result.get('model', platform.config.llm.model),
                        'response': optimized_result['response'],
                        'comparison_mode': True,
                    }
                else:
                    # Tokenomics optimized
                    result = platform.query(query, use_cache=True, use_bandit=True)
            except Exception as e:
                # Handle API errors gracefully
                error_msg = str(e)
                if '429' in error_msg or 'quota' in error_msg.lower():
                    # Return error response for quota issues
                    return jsonify({
                        'error': 'API quota exceeded. The Gemini API quota has been exhausted. Please wait for quota reset or enable billing.',
                        'quota_error': True,
                        'partial_results': current_run if current_run.get('results') else None
                    }), 429
                else:
                    # Re-raise other errors
                    raise
            
            elapsed = (time.time() - start_time) * 1000
            
            # Serialize QueryPlan if present
            plan_data = None
            if result.get('plan'):
                plan = result['plan']
                plan_data = {
                    'complexity': plan.complexity.value if hasattr(plan.complexity, 'value') else str(plan.complexity),
                    'token_budget': plan.token_budget,
                    'model': plan.model,
                    'use_retrieval': plan.use_retrieval,
                    'allocations': [
                        {
                            'component': alloc.component,
                            'tokens': alloc.tokens,
                            'priority': alloc.priority,
                            'utility': alloc.utility,
                        }
                        for alloc in plan.allocations
                    ],
                    'retrieved_context_count': len(plan.retrieved_context),
                    'compressed_prompt': plan.compressed_prompt is not None,
                }
            
            # Get bandit stats if available
            bandit_stats = None
            if result.get('strategy') and hasattr(platform, 'bandit'):
                try:
                    bandit_stats = platform.bandit.stats()
                    # Get specific arm stats
                    arm_id = result.get('strategy')
                    if arm_id and arm_id in bandit_stats.get('arms', {}):
                        bandit_stats['selected_arm'] = {
                            'arm_id': arm_id,
                            'pulls': bandit_stats['arms'][arm_id].get('pulls', 0),
                            'average_reward': bandit_stats['arms'][arm_id].get('average_reward', 0),
                        }
                except:
                    pass
            
            # Determine cache type for display
            cache_type = result.get('cache_type')
            if not cache_type:
                if result['cache_hit']:
                    cache_type = 'hit'  # Generic hit if type not specified
                else:
                    cache_type = 'none'
            
            # Get component-level savings
            # In A/B mode, component_savings are in result['optimized']['component_savings']
            if result.get('comparison_mode') and 'optimized' in result:
                component_savings = result['optimized'].get('component_savings', {})
            else:
                component_savings = result.get('component_savings', {})
            
            query_result = {
                'query_num': i,
                'query': query,
                'response': result.get('response', ''),  # Include full response text
                'tokens_used': result['tokens_used'],
                'input_tokens': result.get('input_tokens', 0),
                'output_tokens': result.get('output_tokens', 0),
                'latency_ms': result['latency_ms'],
                'cache_hit': result['cache_hit'],
                'cache_type': cache_type,
                'similarity': result.get('similarity'),  # Semantic similarity score
                'strategy': result.get('strategy', 'none'),
                'model': result.get('model', platform.config.llm.model),
                'response_length': len(result.get('response', '')),
                'reward': result.get('reward'),  # Bandit reward
                'plan': plan_data,  # Orchestrator plan
                'bandit_stats': bandit_stats,  # Bandit statistics
                'component_savings': component_savings,  # Component-level savings breakdown
                'timestamp': datetime.now().isoformat(),
            }
            
            # Add comparison data if A/B mode
            if result.get('comparison_mode'):
                query_result['comparison'] = {
                    'baseline': result['baseline'],
                    'optimized': result['optimized'],
                    'improvement': result['improvement'],
                }
            
            current_run['results'].append(query_result)
            total_tokens += result['tokens_used']
            total_latency += result['latency_ms']
            if result['cache_hit']:
                cache_hits += 1
            
            tracker.record_query(
                query=query,
                response=result['response'],
                tokens_used=result['tokens_used'],
                latency_ms=result['latency_ms'],
                cache_hit=result['cache_hit'],
                cache_type=result.get('cache_type', 'none'),
                strategy=result.get('strategy', 'none'),
                model=result.get('model', platform.config.llm.model),
            )
            
            # Update global metrics
            if mode == 'ab' and 'baseline_result' in locals():
                update_metrics(result, baseline_result)
            else:
                update_metrics(result)
        
        # Calculate summary
        num_queries = len(current_run['queries'])
        
        # Calculate component-level savings totals
        total_memory_savings = 0
        total_orchestrator_savings = 0
        total_bandit_savings = 0
        
        for r in current_run['results']:
            # In A/B mode, component_savings can be in multiple places
            comp_savings = {}
            if mode == 'ab':
                # Check comparison.optimized first (most reliable)
                if 'comparison' in r and 'optimized' in r['comparison']:
                    comp_savings = r['comparison']['optimized'].get('component_savings', {})
                # Fallback to direct optimized key
                elif 'optimized' in r:
                    comp_savings = r['optimized'].get('component_savings', {})
            else:
                comp_savings = r.get('component_savings', {})
            
            # Only add if comp_savings exists and has values
            if comp_savings:
                total_memory_savings += comp_savings.get('memory_layer', 0) or 0
                total_orchestrator_savings += comp_savings.get('orchestrator', 0) or 0
                total_bandit_savings += comp_savings.get('bandit', 0) or 0
        
        # Calculate A/B comparison improvements if mode is 'ab'
        improvement = None
        component_breakdown = None
        if mode == 'ab':
            baseline_tokens_total = sum(r.get('comparison', {}).get('baseline', {}).get('tokens_used', 0) for r in current_run['results'])
            optimized_tokens_total = total_tokens
            baseline_latency_total = sum(r.get('comparison', {}).get('baseline', {}).get('latency_ms', 0) for r in current_run['results'])
            optimized_latency_total = total_latency
            
            token_savings = baseline_tokens_total - optimized_tokens_total
            token_savings_pct = (token_savings / baseline_tokens_total * 100) if baseline_tokens_total > 0 else 0
            latency_reduction = baseline_latency_total - optimized_latency_total
            latency_reduction_pct = (latency_reduction / baseline_latency_total * 100) if baseline_latency_total > 0 else 0
            
            improvement = {
                'token_savings': token_savings,
                'token_savings_percent': round(token_savings_pct, 2),
                'latency_reduction': round(latency_reduction, 2),
                'latency_reduction_percent': round(latency_reduction_pct, 2),
            }
            
            # Component-level breakdown
            component_breakdown = {
                'memory_layer': {
                    'tokens_saved': total_memory_savings,
                    'percentage': round((total_memory_savings / baseline_tokens_total * 100) if baseline_tokens_total > 0 else 0, 2),
                    'description': 'Tokens saved from cache hits (exact + semantic matches)',
                },
                'orchestrator': {
                    'tokens_saved': total_orchestrator_savings,
                    'percentage': round((total_orchestrator_savings / baseline_tokens_total * 100) if baseline_tokens_total > 0 else 0, 2),
                    'description': 'Tokens saved from better token allocation and compression',
                },
                'bandit': {
                    'tokens_saved': total_bandit_savings,
                    'percentage': round((total_bandit_savings / baseline_tokens_total * 100) if baseline_tokens_total > 0 else 0, 2),
                    'description': 'Tokens saved from optimal strategy selection',
                },
            }
        
        current_run['summary'] = {
            'total_queries': num_queries,
            'total_tokens': total_tokens,
            'average_tokens': total_tokens / num_queries if num_queries > 0 else 0,
            'total_latency_ms': total_latency,
            'average_latency_ms': total_latency / num_queries if num_queries > 0 else 0,
            'cache_hits': cache_hits,
            'cache_hit_rate': (cache_hits / num_queries * 100) if num_queries > 0 else 0,
            'tokens_saved': improvement['token_savings'] if improvement else 0,
            'latency_reduction': improvement['latency_reduction'] if improvement else 0,
            'improvement': improvement,
            'component_breakdown': component_breakdown,  # Component-level savings breakdown
        }
        
        current_run['end_time'] = datetime.now().isoformat()
        current_run['status'] = 'completed'
        
        # Add to history
        runs_history.append(current_run.copy())
        
        # Save to file
        save_run_history()
        
        # Log the response structure for debugging
        logger.info(
            "Experiment completed",
            run_id=current_run['id'],
            num_results=len(current_run['results']),
            mode=current_run['mode'],
            has_summary=bool(current_run.get('summary')),
        )
        
        return jsonify(current_run)
    
    except Exception as e:
        # Catch any other errors
        error_msg = str(e)
        if '429' in error_msg or 'quota' in error_msg.lower():
            return jsonify({
                'error': 'API quota exceeded. Please wait for quota reset or enable billing.',
                'quota_error': True,
                'partial_results': current_run if current_run and current_run.get('results') else None
            }), 429
        else:
            return jsonify({
                'error': error_msg,
                'type': type(e).__name__
            }), 500


@app.route('/api/status')
def get_status():
    """Get current run status."""
    global current_run, platform
    
    # Get environment status
    env_status = get_env_status()
    
    # Try to initialize platform if not already done
    if platform is None:
        try:
            platform = init_platform()
        except ValueError as e:
            # API key missing or invalid
            return jsonify({
                'status': 'error',
                'error': 'API key not configured',
                'error_type': 'missing_api_key',
                'message': str(e),
                'env_status': env_status,
                'current_run': current_run,
                'platform_stats': None,
            }), 200  # Return 200 with error status, not 500
        except Exception as e:
            # Other initialization errors
            return jsonify({
                'status': 'error',
                'error': 'Platform initialization failed',
                'error_type': 'init_error',
                'message': str(e),
                'env_status': env_status,
                'current_run': current_run,
                'platform_stats': None,
            }), 200
    
    # Platform is initialized, get stats
    try:
        stats = platform.get_stats() if platform else {}
        bandit_stats = platform.bandit.stats() if platform else {}
        
        return jsonify({
            'status': 'ok',
            'current_run': current_run,
            'platform_stats': {
                'cache_size': stats.get('memory', {}).get('size', 0),
                'cache_max_size': stats.get('memory', {}).get('max_size', 0),
                'total_tokens_saved': stats.get('memory', {}).get('total_tokens_saved', 0),
                'bandit_pulls': bandit_stats.get('total_pulls', 0),
                'best_strategy': bandit_stats.get('best_strategy', None),
            },
            'env_status': env_status,
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'Failed to get platform stats',
            'error_type': 'stats_error',
            'message': str(e),
            'env_status': env_status,
            'current_run': current_run,
            'platform_stats': None,
        }), 200


@app.route('/api/runs')
def get_runs():
    """Get all historical runs."""
    # Load from file
    load_run_history()
    
    return jsonify({
        'runs': runs_history,
        'total_runs': len(runs_history),
    })


@app.route('/api/runs/<run_id>')
def get_run(run_id):
    """Get specific run details."""
    load_run_history()
    
    for run in runs_history:
        if run['id'] == run_id:
            return jsonify(run)
    
    return jsonify({'error': 'Run not found'}), 404


@app.route('/test-results')
def test_results_page():
    """Test results visualization page."""
    return render_template('test_results.html')


@app.route('/api/test-results')
def get_test_results():
    """Get all test results."""
    try:
        results = scan_test_results()
        return jsonify({
            'success': True,
            'tests': results,
            'total': len(results)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'tests': []
        }), 500


@app.route('/api/test-results/<test_id>')
def get_test_result_details(test_id):
    """Get detailed information for a specific test."""
    try:
        test = get_test_details(test_id)
        if test:
            return jsonify({
                'success': True,
                'test': test
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Test not found'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle single query chat requests with real-time status updates."""
    global platform
    
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Initialize platform if needed
        if platform is None:
            init_platform()
        
        # Run query through platform
        result = platform.query(
            query=query,
            use_cache=True,
            use_bandit=True,
            use_compression=True,
            use_cost_aware_routing=True,
        )
        
        # Serialize QueryPlan if present
        plan_data = None
        if result.get('plan'):
            p = result['plan']
            plan_data = {
                'complexity': p.complexity.value if hasattr(p.complexity, 'value') else str(p.complexity),
                'token_budget': p.token_budget,
                'model': p.model,
                'allocations': [
                    {
                        'component': alloc.component,
                        'tokens': alloc.tokens,
                        'priority': alloc.priority,
                        'utility': alloc.utility,
                    }
                    for alloc in p.allocations
                ],
            }
        
        # Calculate costs
        cost_data = calculate_costs(result)
        
        # Extract latency breakdown
        latency_data = extract_latency_breakdown(result)
        
        # Prepare response
        response = {
            'success': True,
            'response': result.get('response', ''),
            'input_tokens': result.get('input_tokens', 0),
            'output_tokens': result.get('output_tokens', 0),
            'tokens_used': result.get('tokens_used', 0),
            'latency_ms': result.get('latency_ms', 0),
            'cache_hit': result.get('cache_hit', False),
            'cache_type': result.get('cache_type', 'none'),
            'similarity': result.get('similarity'),
            'strategy': result.get('strategy', 'none'),
            'model': result.get('model', platform.config.llm.model),
            'plan': plan_data,
            'costs': cost_data,
            'latency_breakdown': latency_data,
            'component_savings': result.get('component_savings', {}),
            'compression_metrics': result.get('compression_metrics', {}),
            'memory_metrics': result.get('memory_metrics', {}),
            'orchestrator_metrics': result.get('orchestrator_metrics', {}),
            'bandit_metrics': result.get('bandit_metrics', {}),
            'quality_judge': result.get('quality_judge'),
        }
        
        # Update metrics
        update_metrics(result)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error("Chat endpoint error", error=str(e), exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def update_metrics(result: Dict, baseline_result: Optional[Dict] = None):
    """Update global metrics from query result."""
    global metrics_history, platform
    
    metrics_history["total_queries"] += 1
    
    # Update cache metrics
    if result.get("cache_hit"):
        metrics_history["cache_hits"] += 1
    else:
        metrics_history["cache_misses"] += 1
    
    # Update quality scores
    if result.get("quality_judge"):
        quality_score = result["quality_judge"].get("quality_score") or result["quality_judge"].get("confidence", 0)
        metrics_history["quality_scores"].append(quality_score)
        # Keep only last 100 scores
        if len(metrics_history["quality_scores"]) > 100:
            metrics_history["quality_scores"] = metrics_history["quality_scores"][-100:]
    
    # Update component savings (in tokens)
    component_savings = result.get("component_savings", {})
    if component_savings:
        metrics_history["component_savings"]["memory_layer"] += component_savings.get("memory_layer", 0)
        metrics_history["component_savings"]["bandit"] += component_savings.get("bandit", 0)
        metrics_history["component_savings"]["orchestrator"] += component_savings.get("orchestrator", 0)
    
    # Update cost savings if baseline is available
    if baseline_result and platform:
        try:
            savings_data = platform.calculate_real_savings(baseline_result, result)
            metrics_history["total_cost_savings"] += savings_data.get("total_savings", 0)
            metrics_history["total_baseline_cost"] += savings_data.get("baseline_cost", 0)
            metrics_history["total_optimized_cost"] += savings_data.get("optimized_cost", 0)
        except Exception as e:
            logger.warning("Failed to calculate savings for metrics", error=str(e))
    
    # Update strategy distribution
    strategy = result.get("strategy") or result.get("strategy_arm") or "none"
    metrics_history["strategy_distribution"][strategy] = metrics_history["strategy_distribution"].get(strategy, 0) + 1
    
    # Update model distribution
    model = result.get("model", "unknown")
    metrics_history["model_distribution"][model] = metrics_history["model_distribution"].get(model, 0) + 1


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global platform
    
    health_status = {
        "status": "healthy",
        "checks": {},
    }
    
    all_healthy = True
    
    # Check platform initialization
    if platform is None:
        try:
            platform = init_platform()
            health_status["checks"]["platform_initialized"] = True
        except Exception as e:
            health_status["checks"]["platform_initialized"] = False
            health_status["checks"]["platform_error"] = str(e)
            all_healthy = False
    else:
        health_status["checks"]["platform_initialized"] = True
    
    # Check API connectivity (if platform is initialized)
    if platform:
        try:
            # Try a simple token count to verify provider is working
            test_count = platform.orchestrator.count_tokens("test")
            health_status["checks"]["api_connectivity"] = True
            health_status["checks"]["tokenizer_working"] = True
        except Exception as e:
            health_status["checks"]["api_connectivity"] = False
            health_status["checks"]["api_error"] = str(e)
            all_healthy = False
        
        # Check cache status
        try:
            cache_stats = platform.memory.stats()
            health_status["checks"]["cache_status"] = "operational"
            health_status["checks"]["cache_size"] = cache_stats.get("size", 0)
        except Exception as e:
            health_status["checks"]["cache_status"] = "error"
            health_status["checks"]["cache_error"] = str(e)
            all_healthy = False
        
        # Check bandit state
        try:
            bandit_stats = platform.bandit.stats()
            health_status["checks"]["bandit_status"] = "operational"
            health_status["checks"]["bandit_arms"] = len(bandit_stats.get("arms", {}))
        except Exception as e:
            health_status["checks"]["bandit_status"] = "error"
            health_status["checks"]["bandit_error"] = str(e)
            all_healthy = False
    
    if not all_healthy:
        health_status["status"] = "degraded"
        return jsonify(health_status), 503
    
    return jsonify(health_status), 200


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get real-time platform metrics."""
    global platform, metrics_history
    
    try:
        # Calculate cache hit rate
        total_cache_operations = metrics_history["cache_hits"] + metrics_history["cache_misses"]
        cache_hit_rate = (metrics_history["cache_hits"] / total_cache_operations * 100) if total_cache_operations > 0 else 0
        
        # Calculate average quality score
        avg_quality = (
            sum(metrics_history["quality_scores"]) / len(metrics_history["quality_scores"])
            if metrics_history["quality_scores"] else 0
        )
        
        # Calculate cost savings percentage
        savings_percent = (
            (metrics_history["total_cost_savings"] / metrics_history["total_baseline_cost"] * 100)
            if metrics_history["total_baseline_cost"] > 0 else 0
        )
        
        # Get component savings percentages
        total_component_savings = sum(metrics_history["component_savings"].values())
        component_percentages = {}
        if total_component_savings > 0:
            for component, savings in metrics_history["component_savings"].items():
                component_percentages[component] = (savings / total_component_savings * 100)
        else:
            component_percentages = {
                "memory_layer": 0,
                "bandit": 0,
                "orchestrator": 0,
            }
        
        # Get platform stats if available
        platform_stats = {}
        if platform:
            try:
                platform_stats = platform.get_stats()
            except Exception as e:
                logger.warning("Failed to get platform stats", error=str(e))
        
        response = {
            "success": True,
            "metrics": {
                "total_queries": metrics_history["total_queries"],
                "cache_hit_rate": round(cache_hit_rate, 2),
                "cache_hits": metrics_history["cache_hits"],
                "cache_misses": metrics_history["cache_misses"],
                "average_quality_score": round(avg_quality, 3),
                "total_cost_savings": round(metrics_history["total_cost_savings"], 6),
                "total_baseline_cost": round(metrics_history["total_baseline_cost"], 6),
                "total_optimized_cost": round(metrics_history["total_optimized_cost"], 6),
                "savings_percentage": round(savings_percent, 2),
                "component_breakdown": {
                    "memory_layer": {
                        "total_savings": round(metrics_history["component_savings"]["memory_layer"], 2),
                        "percentage": round(component_percentages["memory_layer"], 2),
                    },
                    "bandit": {
                        "total_savings": round(metrics_history["component_savings"]["bandit"], 2),
                        "percentage": round(component_percentages["bandit"], 2),
                    },
                    "orchestrator": {
                        "total_savings": round(metrics_history["component_savings"]["orchestrator"], 2),
                        "percentage": round(component_percentages["orchestrator"], 2),
                    },
                },
                "strategy_distribution": metrics_history["strategy_distribution"],
                "model_distribution": metrics_history["model_distribution"],
                "platform_stats": platform_stats,
            },
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error("Metrics endpoint error", error=str(e), exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/compare', methods=['POST'])
def compare():
    """Run A/B comparison between baseline and Tokenomics."""
    global platform
    
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Initialize platform if needed
        if platform is None:
            init_platform()
        
        # Run baseline query
        baseline_result = platform.query(
            query=query,
            use_cache=False,
            use_bandit=False,
            use_compression=False,
            use_cost_aware_routing=False,
        )
        
        # Run Tokenomics query
        tokenomics_result = platform.query(
            query=query,
            use_cache=True,
            use_bandit=True,
            use_compression=True,
            use_cost_aware_routing=True,
        )
        
        # Serialize plans
        baseline_plan_data = None
        if baseline_result.get('plan'):
            p = baseline_result['plan']
            baseline_plan_data = {
                'complexity': p.complexity.value if hasattr(p.complexity, 'value') else str(p.complexity),
                'token_budget': p.token_budget,
                'model': p.model,
            }
        
        tokenomics_plan_data = None
        if tokenomics_result.get('plan'):
            p = tokenomics_result['plan']
            tokenomics_plan_data = {
                'complexity': p.complexity.value if hasattr(p.complexity, 'value') else str(p.complexity),
                'token_budget': p.token_budget,
                'model': p.model,
            }
        
        # Calculate costs
        baseline_costs = calculate_costs(baseline_result)
        tokenomics_costs = calculate_costs(tokenomics_result)
        
        # Extract decision chain
        tokenomics_decisions = extract_decision_chain(tokenomics_result)
        
        # Calculate comparison metrics
        token_savings = baseline_result.get('tokens_used', 0) - tokenomics_result.get('tokens_used', 0)
        token_savings_percent = (token_savings / baseline_result.get('tokens_used', 1) * 100) if baseline_result.get('tokens_used', 0) > 0 else 0
        
        cost_savings = baseline_costs['baseline_cost'] - tokenomics_costs['actual_cost']
        cost_savings_percent = (cost_savings / baseline_costs['baseline_cost'] * 100) if baseline_costs['baseline_cost'] > 0 else 0
        
        latency_reduction = baseline_result.get('latency_ms', 0) - tokenomics_result.get('latency_ms', 0)
        latency_reduction_percent = (latency_reduction / baseline_result.get('latency_ms', 1) * 100) if baseline_result.get('latency_ms', 0) > 0 else 0
        
        quality_score = None
        if tokenomics_result.get('quality_judge'):
            qj = tokenomics_result['quality_judge']
            if isinstance(qj, dict):
                quality_score = qj.get('score')
            elif hasattr(qj, 'score'):
                quality_score = qj.score
        
        # Prepare response
        response = {
            'success': True,
            'query': query,
            'baseline': {
                'response': baseline_result.get('response', ''),
                'tokens_used': baseline_result.get('tokens_used', 0),
                'input_tokens': baseline_result.get('input_tokens', 0),
                'output_tokens': baseline_result.get('output_tokens', 0),
                'latency_ms': baseline_result.get('latency_ms', 0),
                'model': baseline_result.get('model', 'gpt-4o'),
                'cost': baseline_costs['baseline_cost'],
                'plan': baseline_plan_data,
                'decisions': {
                    'complexity': baseline_plan_data['complexity'] if baseline_plan_data else 'unknown',
                    'model': 'gpt-4o (baseline)',
                    'reasoning': 'No optimization applied - direct LLM call'
                }
            },
            'tokenomics': {
                'response': tokenomics_result.get('response', ''),
                'tokens_used': tokenomics_result.get('tokens_used', 0),
                'input_tokens': tokenomics_result.get('input_tokens', 0),
                'output_tokens': tokenomics_result.get('output_tokens', 0),
                'latency_ms': tokenomics_result.get('latency_ms', 0),
                'model': tokenomics_result.get('model', platform.config.llm.model),
                'cost': tokenomics_costs['actual_cost'],
                'cache_hit': tokenomics_result.get('cache_hit', False),
                'cache_type': tokenomics_result.get('cache_type', 'none'),
                'strategy': tokenomics_result.get('strategy', 'none'),
                'complexity': tokenomics_plan_data['complexity'] if tokenomics_plan_data else 'unknown',
                'component_savings': tokenomics_result.get('component_savings', {}),
                'compression_metrics': tokenomics_result.get('compression_metrics', {}),
                'plan': tokenomics_plan_data,
                'decisions': tokenomics_decisions
            },
            'comparison': {
                'token_savings': token_savings,
                'token_savings_percent': round(token_savings_percent, 2),
                'cost_savings': round(cost_savings, 6),
                'cost_savings_percent': round(cost_savings_percent, 2),
                'latency_reduction': round(latency_reduction, 2),
                'latency_reduction_percent': round(latency_reduction_percent, 2),
                'quality_score': round(quality_score, 3) if quality_score else None
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error("Compare endpoint error", error=str(e), exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats')
def get_stats():
    """Get aggregated statistics."""
    load_run_history()
    
    if not runs_history:
        return jsonify({
            'total_runs': 0,
            'average_token_savings': 0,
            'average_latency_reduction': 0,
            'average_cache_hit_rate': 0,
        })
    
    total_runs = len(runs_history)
    
    # Calculate averages properly
    # For token savings and latency reduction, we want percentage averages from improvement data
    ab_runs = [r for r in runs_history if r.get('mode') == 'ab']
    
    if ab_runs:
        # Calculate average percentages from A/B comparison runs
        token_savings_percents = [
            r.get('summary', {}).get('improvement', {}).get('token_savings_percent', 0)
            for r in ab_runs
            if r.get('summary', {}).get('improvement', {}).get('token_savings_percent') is not None
        ]
        latency_reduction_percents = [
            r.get('summary', {}).get('improvement', {}).get('latency_reduction_percent', 0)
            for r in ab_runs
            if r.get('summary', {}).get('improvement', {}).get('latency_reduction_percent') is not None
        ]
        
        avg_token_savings_pct = sum(token_savings_percents) / len(token_savings_percents) if token_savings_percents else 0
        avg_latency_reduction_pct = sum(latency_reduction_percents) / len(latency_reduction_percents) if latency_reduction_percents else 0
    else:
        avg_token_savings_pct = 0
        avg_latency_reduction_pct = 0
    
    # Cache hit rate is already a percentage
    total_cache_hit_rate = sum(r.get('summary', {}).get('cache_hit_rate', 0) for r in runs_history)
    avg_cache_hit_rate = total_cache_hit_rate / total_runs if total_runs > 0 else 0
    
    return jsonify({
        'total_runs': total_runs,
        'average_token_savings': round(avg_token_savings_pct, 2),
        'average_latency_reduction': round(avg_latency_reduction_pct, 2),
        'average_cache_hit_rate': round(avg_cache_hit_rate, 2),
    })


def save_run_history():
    """Save run history to file."""
    global runs_history
    with open('runs_history.json', 'w') as f:
        json.dump(runs_history, f, indent=2)


def load_run_history():
    """Load run history from file."""
    global runs_history
    if Path('runs_history.json').exists():
        try:
            with open('runs_history.json', 'r') as f:
                runs_history = json.load(f)
        except:
            runs_history = []
    else:
        runs_history = []


# Model pricing per 1M tokens
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gemini-flash": {"input": 0.075, "output": 0.30},
    "gemini-pro": {"input": 1.25, "output": 5.00},
}


def calculate_costs(result, baseline_model="gpt-4o"):
    """
    Calculate actual cost and baseline cost for savings display.
    
    Args:
        result: Platform query result dictionary
        baseline_model: Model to use for baseline comparison (default: gpt-4o)
    
    Returns:
        Dictionary with actual_cost, baseline_cost, net_savings, savings_percent, detailed breakdown
    """
    input_tokens = result.get('input_tokens', 0)
    output_tokens = result.get('output_tokens', 0)
    model = result.get('model', 'gpt-4o-mini')
    
    # Get pricing for actual model
    model_pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
    input_cost = input_tokens / 1_000_000 * model_pricing["input"]
    output_cost = output_tokens / 1_000_000 * model_pricing["output"]
    actual_cost = input_cost + output_cost
    
    # Get pricing for baseline model
    baseline_pricing = MODEL_PRICING.get(baseline_model, MODEL_PRICING["gpt-4o"])
    baseline_input_cost = input_tokens / 1_000_000 * baseline_pricing["input"]
    baseline_output_cost = output_tokens / 1_000_000 * baseline_pricing["output"]
    baseline_cost = baseline_input_cost + baseline_output_cost
    
    net_savings = baseline_cost - actual_cost
    savings_percent = (net_savings / baseline_cost * 100) if baseline_cost > 0 else 0
    
    return {
        "actual_cost": round(actual_cost, 6),
        "baseline_cost": round(baseline_cost, 6),
        "net_savings": round(net_savings, 6),
        "savings_percent": round(savings_percent, 2),
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "baseline_input_cost": round(baseline_input_cost, 6),
        "baseline_output_cost": round(baseline_output_cost, 6),
        "model_pricing": model_pricing,
        "baseline_pricing": baseline_pricing,
    }


def extract_latency_breakdown(result):
    """
    Split total latency into router time vs generation time.
    
    Args:
        result: Platform query result dictionary
    
    Returns:
        Dictionary with router_time_ms, generation_time_ms, total_latency_ms
    """
    total_latency_ms = result.get('latency_ms', 0)
    
    # Try to extract router time from metrics
    router_time_ms = 0
    orchestrator_metrics = result.get('orchestrator_metrics', {})
    bandit_metrics = result.get('bandit_metrics', {})
    
    # Check if we have timing data in metrics
    if isinstance(orchestrator_metrics, dict):
        # Could add timing extraction here if available
        pass
    
    if isinstance(bandit_metrics, dict):
        # Could add timing extraction here if available
        pass
    
    # If no router time found, estimate < 5ms
    if router_time_ms == 0:
        router_time_ms = 2.0  # Conservative estimate
    
    generation_time_ms = max(0, total_latency_ms - router_time_ms)
    
    return {
        "router_time_ms": round(router_time_ms, 2),
        "generation_time_ms": round(generation_time_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
    }


def extract_decision_chain(result):
    """
    Extract full decision chain for transparency.
    
    Args:
        result: Platform query result dictionary
    
    Returns:
        Dictionary with complexity_analysis, routing_decision, cache_decision, compression_decision
    """
    plan = result.get('plan')
    strategy = result.get('strategy', 'none')
    model = result.get('model', 'unknown')
    cache_hit = result.get('cache_hit', False)
    cache_type = result.get('cache_type', 'none')
    compression_metrics = result.get('compression_metrics', {})
    bandit_metrics = result.get('bandit_metrics', {})
    
    # Complexity analysis
    complexity = 'unknown'
    complexity_reasoning = 'Unable to determine complexity'
    if plan:
        complexity = plan.complexity.value if hasattr(plan.complexity, 'value') else str(plan.complexity)
        if complexity == 'simple':
            complexity_reasoning = 'Short query, no complex reasoning required'
        elif complexity == 'medium':
            complexity_reasoning = 'Moderate query length, some reasoning required'
        elif complexity == 'complex':
            complexity_reasoning = 'Long query or complex reasoning required'
    
    # Routing decision
    routing_reasoning = f'{complexity.capitalize()} query'
    if strategy == 'cheap':
        routing_reasoning += ' → cheap strategy selected for cost efficiency'
    elif strategy == 'balanced':
        routing_reasoning += ' → balanced strategy for optimal cost-quality tradeoff'
    elif strategy == 'premium':
        routing_reasoning += ' → premium strategy for maximum quality'
    else:
        routing_reasoning += ' → default strategy'
    
    bandit_confidence = 0.5
    if isinstance(bandit_metrics, dict) and bandit_metrics.get('confidence'):
        bandit_confidence = bandit_metrics['confidence']
    
    # Cache decision
    cache_reasoning = 'No similar queries in cache'
    if cache_hit:
        if cache_type == 'exact':
            cache_reasoning = 'Exact match found in cache'
        elif cache_type == 'semantic_direct':
            similarity = result.get('similarity', 0)
            cache_reasoning = f'Semantic match found (similarity: {similarity:.2f})'
        elif cache_type == 'context':
            similarity = result.get('similarity', 0)
            cache_reasoning = f'Context-enhanced match (similarity: {similarity:.2f})'
    
    # Compression decision
    compression_applied = compression_metrics.get('context_compressed', False) or compression_metrics.get('query_compressed', False)
    compression_reasoning = 'No compression applied'
    if compression_metrics.get('query_compressed'):
        compression_reasoning = 'Query compressed (exceeded threshold)'
    elif compression_metrics.get('context_compressed'):
        ratio = compression_metrics.get('context_compression_ratio', 1.0)
        compression_reasoning = f'Context compressed (ratio: {ratio:.2f})'
    
    return {
        "complexity_analysis": {
            "detected": complexity,
            "reasoning": complexity_reasoning
        },
        "routing_decision": {
            "strategy": strategy,
            "model": model,
            "reasoning": routing_reasoning,
            "bandit_confidence": round(bandit_confidence, 2)
        },
        "cache_decision": {
            "hit": cache_hit,
            "type": cache_type if cache_hit else None,
            "reasoning": cache_reasoning
        },
        "compression_decision": {
            "applied": compression_applied,
            "reasoning": compression_reasoning
        }
    }


if __name__ == '__main__':
    # Load existing history
    load_run_history()
    
    # Run Flask app
    # Use use_reloader=False to prevent constant reloading that interrupts API calls
    # Set use_debugger=True to keep debugger active
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)

