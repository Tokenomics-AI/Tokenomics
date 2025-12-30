#!/usr/bin/env python3
"""Prepare customer support dataset: clean, deduplicate, and split 80:20."""

import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Set
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.training_config import TRAINING_CONFIG


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection."""
    # Convert to lowercase, normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip().lower())
    return text


def clean_query(query: str, product: str = None) -> str:
    """
    Clean a single query.
    
    Args:
        query: Raw query text
        product: Product name to replace {product_purchased} placeholder
    
    Returns:
        Cleaned query text
    """
    if not query or not isinstance(query, str):
        return ""
    
    # Replace {product_purchased} placeholder
    if product:
        query = query.replace("{product_purchased}", product)
    else:
        query = query.replace("{product_purchased}", "product")
    
    # Remove other placeholders that might exist
    query = re.sub(r'\{[^}]+\}', 'product', query)
    
    # Normalize whitespace (multiple spaces/newlines -> single space)
    query = re.sub(r'\s+', ' ', query)
    
    # Remove leading/trailing whitespace
    query = query.strip()
    
    return query


def prepare_dataset():
    """Main function to prepare the dataset."""
    print("=" * 80)
    print("Customer Support Dataset Preparation")
    print("=" * 80)
    print()
    
    config = TRAINING_CONFIG
    dataset_path = Path(config["dataset_path"])
    output_dir = Path(config.get("output_dir", "training_data"))
    output_dir.mkdir(exist_ok=True)
    
    if not dataset_path.exists():
        print(f"✗ ERROR: Dataset file not found: {dataset_path}")
        return 1
    
    print(f"1. Loading dataset from: {dataset_path}")
    print("-" * 80)
    
    queries = []
    seen_normalized = set()
    stats = {
        "total_rows": 0,
        "empty_queries": 0,
        "duplicates": 0,
        "too_short": 0,
        "too_long": 0,
        "after_cleaning": 0,
    }
    
    try:
        with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            
            for idx, row in enumerate(reader, start=1):
                stats["total_rows"] += 1
                
                # Extract query and product
                query_text = row.get(config["query_column"], "").strip()
                product = row.get(config["product_column"], "").strip()
                
                # Clean query
                cleaned = clean_query(query_text, product)
                
                # Check if empty
                if not cleaned or len(cleaned) < config["min_query_length"]:
                    stats["too_short"] += 1
                    continue
                
                # Check if too long
                if len(cleaned) > config["max_query_length"]:
                    stats["too_long"] += 1
                    continue
                
                # Check for duplicates (normalized)
                normalized = normalize_text(cleaned)
                if normalized in seen_normalized:
                    stats["duplicates"] += 1
                    continue
                
                seen_normalized.add(normalized)
                
                # Add to queries
                queries.append({
                    "id": len(queries) + 1,
                    "query": cleaned,
                    "original_ticket_id": row.get("Ticket ID", idx),
                    "product": product or "Unknown",
                    "ticket_type": row.get("Ticket Type", ""),
                    "priority": row.get("Ticket Priority", ""),
                })
                
                stats["after_cleaning"] += 1
                
                if idx % 1000 == 0:
                    print(f"   Processed {idx} rows, {len(queries)} valid queries so far...")
        
        print(f"✓ Loaded {stats['total_rows']} rows")
        print(f"✓ After cleaning: {stats['after_cleaning']} valid queries")
        print(f"   - Empty/too short: {stats['too_short']}")
        print(f"   - Too long: {stats['too_long']}")
        print(f"   - Duplicates: {stats['duplicates']}")
        print()
        
        if len(queries) == 0:
            print("✗ ERROR: No valid queries after cleaning!")
            return 1
        
        # Split 80:20
        print("2. Splitting dataset (80:20 train:test)")
        print("-" * 80)
        
        import random
        random.seed(config["random_seed"])
        random.shuffle(queries)
        
        split_idx = int(len(queries) * config["train_split"])
        training_queries = queries[:split_idx]
        test_queries = queries[split_idx:]
        
        print(f"✓ Training set: {len(training_queries)} queries")
        print(f"✓ Test set: {len(test_queries)} queries")
        print()
        
        # Save training queries
        print("3. Saving prepared datasets")
        print("-" * 80)
        
        training_file = output_dir / "training_queries.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump({
                "queries": training_queries,
                "metadata": {
                    "total_original": stats["total_rows"],
                    "after_cleaning": stats["after_cleaning"],
                    "training_count": len(training_queries),
                    "test_count": len(test_queries),
                    "cleaning_stats": stats,
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved training queries: {training_file}")
        
        test_file = output_dir / "test_queries.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump({
                "queries": test_queries,
                "metadata": {
                    "total_original": stats["total_rows"],
                    "after_cleaning": stats["after_cleaning"],
                    "training_count": len(training_queries),
                    "test_count": len(test_queries),
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved test queries: {test_file}")
        
        stats_file = output_dir / "dataset_stats.json"
        final_stats = {
            "total_original": stats["total_rows"],
            "after_cleaning": stats["after_cleaning"],
            "training_count": len(training_queries),
            "test_count": len(test_queries),
            "cleaning_stats": stats,
            "query_length_stats": {
                "min": min(len(q["query"]) for q in queries),
                "max": max(len(q["query"]) for q in queries),
                "avg": sum(len(q["query"]) for q in queries) / len(queries),
            },
            "product_distribution": dict(Counter(q["product"] for q in queries)),
        }
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved dataset statistics: {stats_file}")
        
        print()
        print("=" * 80)
        print("Dataset Preparation Complete!")
        print("=" * 80)
        print(f"✓ Total queries prepared: {len(queries)}")
        print(f"✓ Training queries: {len(training_queries)}")
        print(f"✓ Test queries: {len(test_queries)}")
        print()
        print("Next step: Run data collection script:")
        print("  python3 scripts/collect_training_data.py")
        
        return 0
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(prepare_dataset())



