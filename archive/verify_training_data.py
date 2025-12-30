#!/usr/bin/env python3
"""
Verify training data format and export to readable format.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

def verify_training_data(db_path: str = "token_prediction_data.db"):
    """Verify and display training data."""
    db = Path(db_path)
    
    if not db.exists():
        print(f"❌ Training database not found: {db_path}")
        return
    
    conn = sqlite3.connect(str(db))
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM token_predictions")
    total = cursor.fetchone()[0]
    print(f"✅ Total training samples: {total}")
    
    if total == 0:
        print("⚠️  No training data collected yet")
        conn.close()
        return
    
    # Get statistics
    cursor.execute("""
        SELECT 
            COUNT(*),
            AVG(actual_output_tokens),
            MIN(actual_output_tokens),
            MAX(actual_output_tokens),
            COUNT(DISTINCT complexity),
            COUNT(DISTINCT model_used)
        FROM token_predictions
    """)
    stats = cursor.fetchone()
    count, avg_tokens, min_tokens, max_tokens, num_complexities, num_models = stats
    
    print(f"\n📊 Statistics:")
    print(f"  Average output tokens: {avg_tokens:.2f}")
    print(f"  Min output tokens: {min_tokens}")
    print(f"  Max output tokens: {max_tokens}")
    print(f"  Unique complexities: {num_complexities}")
    print(f"  Unique models: {num_models}")
    
    # Get complexity distribution
    cursor.execute("""
        SELECT complexity, COUNT(*) as count
        FROM token_predictions
        GROUP BY complexity
        ORDER BY count DESC
    """)
    print(f"\n📈 Complexity Distribution:")
    for complexity, count in cursor.fetchall():
        print(f"  {complexity}: {count} samples")
    
    # Get model distribution
    cursor.execute("""
        SELECT model_used, COUNT(*) as count
        FROM token_predictions
        WHERE model_used IS NOT NULL
        GROUP BY model_used
        ORDER BY count DESC
    """)
    print(f"\n🤖 Model Distribution:")
    for model, count in cursor.fetchall():
        print(f"  {model}: {count} samples")
    
    # Show sample data
    cursor.execute("""
        SELECT query, query_length, complexity, embedding_vector, 
               predicted_tokens, actual_output_tokens, model_used, timestamp
        FROM token_predictions
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    
    print(f"\n📝 Sample Data (Latest 10):")
    print("=" * 100)
    for row in cursor.fetchall():
        query, qlen, complexity, emb_json, pred, actual, model, ts = row
        emb = json.loads(emb_json) if emb_json else None
        emb_preview = f"{emb[:3]}..." if emb and len(emb) > 3 else "None"
        
        print(f"\nQuery: {query[:60]}...")
        print(f"  Length: {qlen} tokens | Complexity: {complexity}")
        print(f"  Predicted: {pred} | Actual: {actual} | Model: {model or 'N/A'}")
        print(f"  Embedding: {emb_preview} | Timestamp: {ts}")
    
    # Export to JSON for inspection
    cursor.execute("""
        SELECT query, query_length, complexity, embedding_vector,
               predicted_tokens, actual_output_tokens, model_used, timestamp
        FROM token_predictions
        ORDER BY timestamp DESC
    """)
    
    samples = []
    for row in cursor.fetchall():
        query, qlen, complexity, emb_json, pred, actual, model, ts = row
        emb = json.loads(emb_json) if emb_json else None
        samples.append({
            "query": query,
            "query_length": qlen,
            "complexity": complexity,
            "embedding_vector": emb,
            "predicted_tokens": pred,
            "actual_output_tokens": actual,
            "model_used": model,
            "timestamp": ts,
        })
    
    export_file = f"training_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(export_file, 'w') as f:
        json.dump({
            "metadata": {
                "total_samples": total,
                "export_timestamp": datetime.now().isoformat(),
                "format_version": "1.0",
            },
            "samples": samples,
        }, f, indent=2)
    
    print(f"\n💾 Exported to: {export_file}")
    print(f"   Format: JSON with all fields properly structured")
    
    conn.close()
    print("\n✅ Training data verification complete!")

if __name__ == "__main__":
    verify_training_data()






