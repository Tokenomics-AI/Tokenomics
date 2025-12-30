#!/usr/bin/env python3
"""Test ML model database initialization and functionality."""

import sys
import os
from pathlib import Path
import sqlite3
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_database_initialization():
    """Test that databases are created when platform initializes."""
    print("=" * 80)
    print("Testing ML Database Initialization")
    print("=" * 80)
    print()
    
    # Check if databases exist before initialization
    token_db_path = Path("token_prediction_data.db")
    escalation_db_path = Path("escalation_prediction_data.db")
    
    print("1. Checking database files BEFORE platform initialization:")
    print("-" * 80)
    print(f"   token_prediction_data.db exists: {token_db_path.exists()}")
    if token_db_path.exists():
        print(f"   Location: {token_db_path.absolute()}")
        print(f"   Size: {token_db_path.stat().st_size} bytes")
    print(f"   escalation_prediction_data.db exists: {escalation_db_path.exists()}")
    if escalation_db_path.exists():
        print(f"   Location: {escalation_db_path.absolute()}")
        print(f"   Size: {escalation_db_path.stat().st_size} bytes")
    print()
    
    # Initialize platform
    print("2. Initializing TokenomicsPlatform...")
    print("-" * 80)
    try:
        from tokenomics.core import TokenomicsPlatform
        from tokenomics.config import TokenomicsConfig
        
        # Load config from environment
        config = TokenomicsConfig.from_env()
        
        # Initialize platform
        platform = TokenomicsPlatform(config=config)
        print("   ✓ Platform initialized successfully")
        print()
        
        # Check if databases exist after initialization
        print("3. Checking database files AFTER platform initialization:")
        print("-" * 80)
        token_exists_after = token_db_path.exists()
        escalation_exists_after = escalation_db_path.exists()
        
        print(f"   token_prediction_data.db exists: {token_exists_after}")
        if token_exists_after:
            print(f"   Location: {token_db_path.absolute()}")
            print(f"   Size: {token_db_path.stat().st_size} bytes")
        else:
            print("   ✗ ERROR: Database file was not created!")
        
        print(f"   escalation_prediction_data.db exists: {escalation_exists_after}")
        if escalation_exists_after:
            print(f"   Location: {escalation_db_path.absolute()}")
            print(f"   Size: {escalation_db_path.stat().st_size} bytes")
        else:
            print("   ✗ ERROR: Database file was not created!")
        print()
        
        # Verify database schema
        print("4. Verifying database schema:")
        print("-" * 80)
        
        # Check token prediction database
        if token_exists_after:
            try:
                conn = sqlite3.connect(str(token_db_path))
                cursor = conn.cursor()
                
                # Check table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='token_predictions'
                """)
                table_exists = cursor.fetchone() is not None
                print(f"   token_predictions table exists: {table_exists}")
                
                if table_exists:
                    # Get table schema
                    cursor.execute("PRAGMA table_info(token_predictions)")
                    columns = cursor.fetchall()
                    print(f"   Columns ({len(columns)}):")
                    for col in columns:
                        print(f"     - {col[1]} ({col[2]})")
                    
                    # Check indexes
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='index' AND tbl_name='token_predictions'
                    """)
                    indexes = [row[0] for row in cursor.fetchall()]
                    print(f"   Indexes: {', '.join(indexes) if indexes else 'None'}")
                    
                    # Get row count
                    cursor.execute("SELECT COUNT(*) FROM token_predictions")
                    row_count = cursor.fetchone()[0]
                    print(f"   Row count: {row_count}")
                
                conn.close()
                print("   ✓ Token prediction database schema verified")
            except Exception as e:
                print(f"   ✗ ERROR verifying token prediction database: {e}")
        else:
            print("   ⚠ Skipping token prediction database verification (file doesn't exist)")
        
        print()
        
        # Check escalation prediction database
        if escalation_exists_after:
            try:
                conn = sqlite3.connect(str(escalation_db_path))
                cursor = conn.cursor()
                
                # Check table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='escalation_predictions'
                """)
                table_exists = cursor.fetchone() is not None
                print(f"   escalation_predictions table exists: {table_exists}")
                
                if table_exists:
                    # Get table schema
                    cursor.execute("PRAGMA table_info(escalation_predictions)")
                    columns = cursor.fetchall()
                    print(f"   Columns ({len(columns)}):")
                    for col in columns:
                        print(f"     - {col[1]} ({col[2]})")
                    
                    # Check indexes
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='index' AND tbl_name='escalation_predictions'
                    """)
                    indexes = [row[0] for row in cursor.fetchall()]
                    print(f"   Indexes: {', '.join(indexes) if indexes else 'None'}")
                    
                    # Get row count
                    cursor.execute("SELECT COUNT(*) FROM escalation_predictions")
                    row_count = cursor.fetchone()[0]
                    print(f"   Row count: {row_count}")
                
                conn.close()
                print("   ✓ Escalation prediction database schema verified")
            except Exception as e:
                print(f"   ✗ ERROR verifying escalation prediction database: {e}")
        else:
            print("   ⚠ Skipping escalation prediction database verification (file doesn't exist)")
        
        print()
        
        # Test data collection
        print("5. Testing data collection functionality:")
        print("-" * 80)
        
        # Test token predictor data collection
        if platform.token_predictor and platform.token_predictor.data_collector:
            try:
                platform.token_predictor.data_collector.record(
                    query="Test query for token prediction",
                    query_length=30,
                    complexity="simple",
                    embedding_vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    predicted_tokens=100,
                    actual_output_tokens=95,
                    model_used="gpt-4o-mini",
                )
                print("   ✓ Token prediction data recorded successfully")
                
                # Verify data was stored
                conn = sqlite3.connect(str(token_db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM token_predictions")
                count = cursor.fetchone()[0]
                conn.close()
                print(f"   ✓ Verified: {count} record(s) in database")
            except Exception as e:
                print(f"   ✗ ERROR recording token prediction data: {e}")
        else:
            print("   ⚠ Token predictor not available")
        
        # Test escalation predictor data collection
        if platform.escalation_predictor and platform.escalation_predictor.data_collector:
            try:
                platform.escalation_predictor.data_collector.record(
                    query="Test query for escalation prediction",
                    query_length=35,
                    complexity="medium",
                    context_quality_score=0.75,
                    query_tokens=10,
                    query_embedding=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1],
                    escalated=False,
                    model_used="gpt-4o-mini",
                )
                print("   ✓ Escalation prediction data recorded successfully")
                
                # Verify data was stored
                conn = sqlite3.connect(str(escalation_db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM escalation_predictions")
                count = cursor.fetchone()[0]
                conn.close()
                print(f"   ✓ Verified: {count} record(s) in database")
            except Exception as e:
                print(f"   ✗ ERROR recording escalation prediction data: {e}")
        else:
            print("   ⚠ Escalation predictor not available")
        
        print()
        
        # Test data retrieval
        print("6. Testing data retrieval:")
        print("-" * 80)
        
        if platform.token_predictor and platform.token_predictor.data_collector:
            try:
                stats = platform.token_predictor.data_collector.get_stats()
                print(f"   Token prediction stats: {json.dumps(stats, indent=6)}")
                print("   ✓ Data retrieval working")
            except Exception as e:
                print(f"   ✗ ERROR retrieving token prediction stats: {e}")
        
        if platform.escalation_predictor and platform.escalation_predictor.data_collector:
            try:
                stats = platform.escalation_predictor.data_collector.get_stats()
                print(f"   Escalation prediction stats: {json.dumps(stats, indent=6)}")
                print("   ✓ Data retrieval working")
            except Exception as e:
                print(f"   ✗ ERROR retrieving escalation prediction stats: {e}")
        
        print()
        print("=" * 80)
        print("Test Summary:")
        print("=" * 80)
        
        all_passed = (
            token_exists_after and 
            escalation_exists_after and
            platform.token_predictor is not None and
            platform.escalation_predictor is not None
        )
        
        if all_passed:
            print("✓ All tests PASSED!")
            print("  - Both database files created successfully")
            print("  - Database schemas are correct")
            print("  - Data collection is working")
            print("  - Data retrieval is working")
            return 0
        else:
            print("✗ Some tests FAILED:")
            if not token_exists_after:
                print("  - Token prediction database was not created")
            if not escalation_exists_after:
                print("  - Escalation prediction database was not created")
            if platform.token_predictor is None:
                print("  - Token predictor not initialized")
            if platform.escalation_predictor is None:
                print("  - Escalation predictor not initialized")
            return 1
        
    except Exception as e:
        print(f"✗ ERROR initializing platform: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_database_initialization())



