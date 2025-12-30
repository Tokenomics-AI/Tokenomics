#!/usr/bin/env python3
"""Collect training data by running queries through the platform."""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.training_config import TRAINING_CONFIG


def load_training_queries() -> List[Dict]:
    """Load training queries from JSON file."""
    output_dir = Path(TRAINING_CONFIG.get("output_dir", "training_data"))
    training_file = output_dir / "training_queries.json"
    
    if not training_file.exists():
        print(f"✗ ERROR: Training queries file not found: {training_file}")
        print("   Run data preparation script first:")
        print("   python3 scripts/prepare_customer_support_dataset.py")
        return []
    
    with open(training_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get("queries", [])


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint if exists."""
    output_dir = Path(TRAINING_CONFIG.get("output_dir", "training_data"))
    checkpoint_file = output_dir / "checkpoint.json"
    
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_data: Dict):
    """Save checkpoint."""
    output_dir = Path(TRAINING_CONFIG.get("output_dir", "training_data"))
    output_dir.mkdir(exist_ok=True)
    checkpoint_file = output_dir / "checkpoint.json"
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2)


def get_database_stats(platform) -> Dict:
    """Get current database statistics."""
    if not platform or not hasattr(platform, 'token_predictor'):
        return {}
    
    try:
        if platform.token_predictor and platform.token_predictor.data_collector:
            stats = platform.token_predictor.data_collector.get_stats()
            return {
                "token_predictions": stats.get("token_prediction", {}).get("total_samples", 0),
                "escalation_predictions": stats.get("escalation_prediction", {}).get("total_samples", 0),
                "complexity_predictions": stats.get("complexity_prediction", {}).get("total_samples", 0),
            }
    except Exception as e:
        print(f"   Warning: Could not get database stats: {e}")
    
    return {}


def collect_training_data(sample_size: Optional[int] = None, resume: bool = True):
    """Collect training data by running queries through platform."""
    print("=" * 80)
    print("Training Data Collection")
    print("=" * 80)
    print()
    
    config = TRAINING_CONFIG
    sample_size = sample_size or config["sample_size"]
    checkpoint_interval = config["checkpoint_interval"]
    batch_size = config["batch_size"]
    
    # Load queries
    print("1. Loading training queries")
    print("-" * 80)
    queries = load_training_queries()
    
    if not queries:
        return 1
    
    print(f"✓ Loaded {len(queries)} training queries")
    
    # Check for checkpoint
    checkpoint = None
    start_index = 0
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            start_index = checkpoint.get("last_processed_index", 0)
            print(f"✓ Found checkpoint: resuming from index {start_index}")
            print(f"   Previous run: {checkpoint.get('queries_run', 0)} queries processed")
        else:
            print("✓ No checkpoint found, starting fresh")
    else:
        print("✓ Starting fresh (resume disabled)")
    
    print()
    
    # Sample queries if needed
    if len(queries) > sample_size:
        import random
        random.seed(config["random_seed"])
        if start_index == 0:
            # Only sample if starting fresh
            queries = random.sample(queries, sample_size)
            print(f"✓ Sampled {sample_size} queries for data collection")
        else:
            # If resuming, use remaining queries
            queries = queries[start_index:]
            print(f"✓ Using remaining {len(queries)} queries from checkpoint")
    else:
        if start_index > 0:
            queries = queries[start_index:]
        print(f"✓ Using all {len(queries)} queries")
    
    print()
    
    # Initialize platform
    print("2. Initializing TokenomicsPlatform")
    print("-" * 80)
    try:
        from tokenomics.core import TokenomicsPlatform
        from tokenomics.config import TokenomicsConfig
        
        platform_config = TokenomicsConfig.from_env()
        platform = TokenomicsPlatform(config=platform_config)
        print("✓ Platform initialized")
        print(f"   - Token predictor: {'✓' if platform.token_predictor else '✗'}")
        print(f"   - Escalation predictor: {'✓' if platform.escalation_predictor else '✗'}")
        print(f"   - Complexity classifier: {'✓' if platform.complexity_classifier else '✗'}")
    except Exception as e:
        print(f"✗ ERROR: Failed to initialize platform: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # Get initial stats
    initial_stats = get_database_stats(platform)
    print("3. Current database statistics")
    print("-" * 80)
    print(f"   Token predictions: {initial_stats.get('token_predictions', 0)}")
    print(f"   Escalation predictions: {initial_stats.get('escalation_predictions', 0)}")
    print(f"   Complexity predictions: {initial_stats.get('complexity_predictions', 0)}")
    print()
    
    # Run queries
    print("4. Running queries through platform")
    print("-" * 80)
    
    total_queries = len(queries)
    successful = 0
    failed = 0
    start_time = time.time()
    
    try:
        for idx, query_data in enumerate(queries, start=1):
            query_text = query_data["query"]
            query_id = query_data.get("id", idx)
            
            try:
                # Run query through platform
                # Platform automatically records:
                # - Token prediction (actual output tokens)
                # - Escalation outcome (if cascading was used)
                # - Complexity prediction (with heuristic as ground truth)
                result = platform.query(
                    query=query_text,
                    use_cache=False,  # Disable cache to get fresh data
                    use_bandit=True,
                    use_compression=True,
                )
                
                successful += 1
                
                # Log progress
                if idx % batch_size == 0 or idx == total_queries:
                    elapsed = time.time() - start_time
                    rate = idx / elapsed if elapsed > 0 else 0
                    remaining = (total_queries - idx) / rate if rate > 0 else 0
                    
                    current_stats = get_database_stats(platform)
                    print(f"   Progress: {idx}/{total_queries} ({idx*100//total_queries}%) | "
                          f"Success: {successful} | Failed: {failed} | "
                          f"Rate: {rate:.1f} queries/sec | "
                          f"ETA: {remaining:.0f}s")
                    print(f"   Data collected: Token={current_stats.get('token_predictions', 0)}, "
                          f"Escalation={current_stats.get('escalation_predictions', 0)}, "
                          f"Complexity={current_stats.get('complexity_predictions', 0)}")
                
                # Save checkpoint
                if idx % checkpoint_interval == 0:
                    current_stats = get_database_stats(platform)
                    checkpoint_data = {
                        "last_processed_index": start_index + idx,
                        "queries_run": start_index + idx,
                        "successful": successful,
                        "failed": failed,
                        "timestamp": datetime.now().isoformat(),
                        "data_stats": current_stats,
                    }
                    save_checkpoint(checkpoint_data)
                    print(f"   ✓ Checkpoint saved at query {idx}")
                
            except Exception as e:
                failed += 1
                print(f"   ✗ Query {idx} failed: {str(e)[:100]}")
                # Continue with next query
                continue
        
        print()
        print("✓ Data collection complete!")
        
    except KeyboardInterrupt:
        print()
        print("⚠ Interrupted by user")
        # Save checkpoint before exiting
        current_stats = get_database_stats(platform)
        checkpoint_data = {
            "last_processed_index": start_index + idx,
            "queries_run": start_index + idx,
            "successful": successful,
            "failed": failed,
            "timestamp": datetime.now().isoformat(),
            "data_stats": current_stats,
            "interrupted": True,
        }
        save_checkpoint(checkpoint_data)
        print(f"✓ Checkpoint saved. Resume with: python3 scripts/collect_training_data.py")
    
    # Final stats
    print()
    print("5. Final statistics")
    print("-" * 80)
    
    final_stats = get_database_stats(platform)
    elapsed = time.time() - start_time
    
    print(f"   Queries processed: {successful + failed}")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Time elapsed: {elapsed:.1f} seconds")
    print(f"   Average rate: {(successful + failed) / elapsed:.2f} queries/sec")
    print()
    print("   Data collected:")
    print(f"   - Token predictions: {final_stats.get('token_predictions', 0)} "
          f"(+{final_stats.get('token_predictions', 0) - initial_stats.get('token_predictions', 0)})")
    print(f"   - Escalation predictions: {final_stats.get('escalation_predictions', 0)} "
          f"(+{final_stats.get('escalation_predictions', 0) - initial_stats.get('escalation_predictions', 0)})")
    print(f"   - Complexity predictions: {final_stats.get('complexity_predictions', 0)} "
          f"(+{final_stats.get('complexity_predictions', 0) - initial_stats.get('complexity_predictions', 0)})")
    print()
    
    # Check if ready for training
    print("6. Training readiness check")
    print("-" * 80)
    
    min_samples = config["min_samples"]
    token_ready = final_stats.get('token_predictions', 0) >= min_samples["token_predictor"]
    escalation_ready = final_stats.get('escalation_predictions', 0) >= min_samples["escalation_predictor"]
    complexity_ready = final_stats.get('complexity_predictions', 0) >= min_samples["complexity_classifier"]
    
    print(f"   Token Predictor: {final_stats.get('token_predictions', 0)}/{min_samples['token_predictor']} "
          f"{'✓ READY' if token_ready else '✗ NEEDS MORE DATA'}")
    print(f"   Escalation Predictor: {final_stats.get('escalation_predictions', 0)}/{min_samples['escalation_predictor']} "
          f"{'✓ READY' if escalation_ready else '✗ NEEDS MORE DATA'}")
    print(f"   Complexity Classifier: {final_stats.get('complexity_predictions', 0)}/{min_samples['complexity_classifier']} "
          f"{'✓ READY' if complexity_ready else '✗ NEEDS MORE DATA'}")
    print()
    
    if token_ready and escalation_ready and complexity_ready:
        print("✓ All models ready for training!")
        print("   Next step: python3 scripts/train_ml_models.py")
    else:
        print("⚠ Some models need more data. Consider:")
        print("   - Running more queries")
        print("   - Adjusting min_samples in config")
        print("   - Resuming data collection: python3 scripts/collect_training_data.py")
    
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect training data by running queries through platform")
    parser.add_argument("--sample-size", type=int, help="Number of queries to run (default: from config)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    args = parser.parse_args()
    
    sys.exit(collect_training_data(
        sample_size=args.sample_size,
        resume=not args.no_resume
    ))



