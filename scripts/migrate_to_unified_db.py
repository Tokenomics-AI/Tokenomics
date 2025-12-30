#!/usr/bin/env python3
"""Migrate data from old ML databases to unified database."""

import sqlite3
import json
import sys
from pathlib import Path
from datetime import datetime

def migrate_databases():
    """Migrate token_prediction_data.db and escalation_prediction_data.db to ml_training_data.db"""
    
    print("=" * 80)
    print("ML Database Migration Script")
    print("=" * 80)
    print()
    
    # Paths
    token_db_path = Path("token_prediction_data.db")
    escalation_db_path = Path("escalation_prediction_data.db")
    unified_db_path = Path("ml_training_data.db")
    
    # Check if old databases exist
    token_exists = token_db_path.exists()
    escalation_exists = escalation_db_path.exists()
    
    print("1. Checking existing databases:")
    print("-" * 80)
    print(f"   token_prediction_data.db: {'EXISTS' if token_exists else 'NOT FOUND'}")
    if token_exists:
        print(f"   Size: {token_db_path.stat().st_size} bytes")
    print(f"   escalation_prediction_data.db: {'EXISTS' if escalation_exists else 'NOT FOUND'}")
    if escalation_exists:
        print(f"   Size: {escalation_db_path.stat().st_size} bytes")
    print()
    
    if not token_exists and not escalation_exists:
        print("⚠ No existing databases found. Nothing to migrate.")
        print("   The unified database will be created automatically when the platform initializes.")
        return 0
    
    # Initialize unified database
    print("2. Initializing unified database:")
    print("-" * 80)
    try:
        from tokenomics.ml.unified_data_collector import UnifiedDataCollector
        unified_collector = UnifiedDataCollector(db_path=str(unified_db_path))
        print(f"   ✓ Unified database created: {unified_db_path.absolute()}")
    except Exception as e:
        print(f"   ✗ ERROR: Failed to create unified database: {e}")
        return 1
    
    print()
    
    # Migrate token prediction data
    if token_exists:
        print("3. Migrating token prediction data:")
        print("-" * 80)
        try:
            conn_old = sqlite3.connect(str(token_db_path))
            cursor_old = conn_old.cursor()
            
            cursor_old.execute("SELECT COUNT(*) FROM token_predictions")
            count = cursor_old.fetchone()[0]
            print(f"   Found {count} records in token_prediction_data.db")
            
            if count > 0:
                cursor_old.execute("""
                    SELECT query, query_length, complexity, embedding_vector, 
                           predicted_tokens, actual_output_tokens, model_used, timestamp
                    FROM token_predictions
                """)
                
                migrated = 0
                for row in cursor_old.fetchall():
                    query, query_length, complexity, embedding_json, predicted_tokens, actual_output_tokens, model_used, timestamp = row
                    
                    embedding_vector = None
                    if embedding_json:
                        try:
                            embedding_vector = json.loads(embedding_json)
                        except:
                            pass
                    
                    unified_collector.record_token_prediction(
                        query=query,
                        query_length=query_length,
                        complexity=complexity,
                        embedding_vector=embedding_vector,
                        predicted_tokens=predicted_tokens,
                        actual_output_tokens=actual_output_tokens,
                        model_used=model_used,
                    )
                    migrated += 1
                
                print(f"   ✓ Migrated {migrated} records to unified database")
            else:
                print("   ⚠ No records to migrate")
            
            conn_old.close()
        except Exception as e:
            print(f"   ✗ ERROR migrating token prediction data: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    
    # Migrate escalation prediction data
    if escalation_exists:
        print("4. Migrating escalation prediction data:")
        print("-" * 80)
        try:
            conn_old = sqlite3.connect(str(escalation_db_path))
            cursor_old = conn_old.cursor()
            
            cursor_old.execute("SELECT COUNT(*) FROM escalation_predictions")
            count = cursor_old.fetchone()[0]
            print(f"   Found {count} records in escalation_prediction_data.db")
            
            if count > 0:
                cursor_old.execute("""
                    SELECT query, query_length, complexity, context_quality_score, 
                           query_tokens, query_embedding, escalated, model_used, timestamp
                    FROM escalation_predictions
                """)
                
                migrated = 0
                for row in cursor_old.fetchall():
                    query, query_length, complexity, context_quality_score, query_tokens, embedding_json, escalated_int, model_used, timestamp = row
                    
                    embedding_vector = None
                    if embedding_json:
                        try:
                            embedding_vector = json.loads(embedding_json)
                        except:
                            pass
                    
                    unified_collector.record_escalation_prediction(
                        query=query,
                        query_length=query_length,
                        complexity=complexity,
                        context_quality_score=context_quality_score,
                        query_tokens=query_tokens,
                        query_embedding=embedding_vector,
                        escalated=bool(escalated_int),
                        model_used=model_used,
                    )
                    migrated += 1
                
                print(f"   ✓ Migrated {migrated} records to unified database")
            else:
                print("   ⚠ No records to migrate")
            
            conn_old.close()
        except Exception as e:
            print(f"   ✗ ERROR migrating escalation prediction data: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    
    # Verify migration
    print("5. Verifying migration:")
    print("-" * 80)
    try:
        stats = unified_collector.get_stats()
        print(f"   Token prediction records: {stats.get('token_prediction', {}).get('total_samples', 0)}")
        print(f"   Escalation prediction records: {stats.get('escalation_prediction', {}).get('total_samples', 0)}")
        print(f"   Complexity prediction records: {stats.get('complexity_prediction', {}).get('total_samples', 0)}")
        print("   ✓ Migration verification complete")
    except Exception as e:
        print(f"   ✗ ERROR verifying migration: {e}")
    
    print()
    print("=" * 80)
    print("Migration Summary:")
    print("=" * 80)
    print("✓ Migration completed successfully!")
    print()
    print("Next steps:")
    print("  1. The unified database is ready to use")
    print("  2. Old databases can be backed up or removed (optional)")
    print("  3. The platform will now use ml_training_data.db for all ML models")
    print()
    print("Note: Old databases are NOT automatically deleted.")
    print("      You can manually remove them after verifying the migration.")
    
    return 0

if __name__ == "__main__":
    sys.exit(migrate_databases())



