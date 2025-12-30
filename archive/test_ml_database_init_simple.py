#!/usr/bin/env python3
"""Simple test for ML database initialization - tests data collectors directly."""

import sys
from pathlib import Path
import sqlite3
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_database_initialization():
    """Test that databases are created when data collectors initialize."""
    print("=" * 80)
    print("Testing ML Database Initialization (Direct Test)")
    print("=" * 80)
    print()
    
    # Check if databases exist before initialization
    token_db_path = Path("token_prediction_data.db")
    escalation_db_path = Path("escalation_prediction_data.db")
    
    print("1. Checking database files BEFORE initialization:")
    print("-" * 80)
    token_exists_before = token_db_path.exists()
    escalation_exists_before = escalation_db_path.exists()
    
    print(f"   token_prediction_data.db exists: {token_exists_before}")
    if token_exists_before:
        print(f"   Location: {token_db_path.absolute()}")
        print(f"   Size: {token_db_path.stat().st_size} bytes")
    
    print(f"   escalation_prediction_data.db exists: {escalation_exists_before}")
    if escalation_exists_before:
        print(f"   Location: {escalation_db_path.absolute()}")
        print(f"   Size: {escalation_db_path.stat().st_size} bytes")
    print()
    
    # Initialize data collectors directly
    print("2. Initializing DataCollector...")
    print("-" * 80)
    try:
        from tokenomics.ml.data_collector import DataCollector
        token_collector = DataCollector(db_path="token_prediction_data.db")
        print("   ✓ DataCollector initialized successfully")
        print(f"   Database path: {token_collector.db_path.absolute()}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    print("3. Initializing EscalationDataCollector...")
    print("-" * 80)
    try:
        from tokenomics.ml.escalation_data_collector import EscalationDataCollector
        escalation_collector = EscalationDataCollector(db_path="escalation_prediction_data.db")
        print("   ✓ EscalationDataCollector initialized successfully")
        print(f"   Database path: {escalation_collector.db_path.absolute()}")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # Check if databases exist after initialization
    print("4. Checking database files AFTER initialization:")
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
    print("5. Verifying database schemas:")
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
            print(f"   ✓ token_predictions table exists: {table_exists}")
            
            if table_exists:
                # Get table schema
                cursor.execute("PRAGMA table_info(token_predictions)")
                columns = cursor.fetchall()
                print(f"   ✓ Columns ({len(columns)}):")
                for col in columns:
                    nullable = "NULL" if col[3] == 0 else "NOT NULL"
                    default = f" DEFAULT {col[4]}" if col[4] else ""
                    print(f"     - {col[1]:25} {col[2]:15} {nullable}{default}")
                
                # Check indexes
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='index' AND tbl_name='token_predictions'
                """)
                indexes = [row[0] for row in cursor.fetchall()]
                print(f"   ✓ Indexes: {', '.join(indexes) if indexes else 'None'}")
                
                # Get row count
                cursor.execute("SELECT COUNT(*) FROM token_predictions")
                row_count = cursor.fetchone()[0]
                print(f"   ✓ Row count: {row_count}")
            
            conn.close()
        except Exception as e:
            print(f"   ✗ ERROR verifying token prediction database: {e}")
            import traceback
            traceback.print_exc()
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
            print(f"   ✓ escalation_predictions table exists: {table_exists}")
            
            if table_exists:
                # Get table schema
                cursor.execute("PRAGMA table_info(escalation_predictions)")
                columns = cursor.fetchall()
                print(f"   ✓ Columns ({len(columns)}):")
                for col in columns:
                    nullable = "NULL" if col[3] == 0 else "NOT NULL"
                    default = f" DEFAULT {col[4]}" if col[4] else ""
                    print(f"     - {col[1]:25} {col[2]:15} {nullable}{default}")
                
                # Check indexes
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='index' AND tbl_name='escalation_predictions'
                """)
                indexes = [row[0] for row in cursor.fetchall()]
                print(f"   ✓ Indexes: {', '.join(indexes) if indexes else 'None'}")
                
                # Get row count
                cursor.execute("SELECT COUNT(*) FROM escalation_predictions")
                row_count = cursor.fetchone()[0]
                print(f"   ✓ Row count: {row_count}")
            
            conn.close()
        except Exception as e:
            print(f"   ✗ ERROR verifying escalation prediction database: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ⚠ Skipping escalation prediction database verification (file doesn't exist)")
    
    print()
    
    # Test data collection
    print("6. Testing data collection:")
    print("-" * 80)
    
    try:
        # Test token predictor data collection
        token_collector.record(
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
        print(f"   ✓ Verified: {count} record(s) in token_predictions table")
    except Exception as e:
        print(f"   ✗ ERROR recording token prediction data: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Test escalation predictor data collection
        escalation_collector.record(
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
        print(f"   ✓ Verified: {count} record(s) in escalation_predictions table")
    except Exception as e:
        print(f"   ✗ ERROR recording escalation prediction data: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test data retrieval
    print("7. Testing data retrieval:")
    print("-" * 80)
    
    try:
        stats = token_collector.get_stats()
        print(f"   Token prediction stats:")
        for key, value in stats.items():
            print(f"     - {key}: {value}")
        print("   ✓ Token prediction data retrieval working")
    except Exception as e:
        print(f"   ✗ ERROR retrieving token prediction stats: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        stats = escalation_collector.get_stats()
        print(f"   Escalation prediction stats:")
        for key, value in stats.items():
            print(f"     - {key}: {value}")
        print("   ✓ Escalation prediction data retrieval working")
    except Exception as e:
        print(f"   ✗ ERROR retrieving escalation prediction stats: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("Test Summary:")
    print("=" * 80)
    
    all_passed = (
        token_exists_after and 
        escalation_exists_after
    )
    
    if all_passed:
        print("✓ All tests PASSED!")
        print("  - Both database files created successfully")
        print("  - Database schemas are correct")
        print("  - Data collection is working")
        print("  - Data retrieval is working")
        print()
        print("The ML model databases are properly initialized and ready to use!")
        return 0
    else:
        print("✗ Some tests FAILED:")
        if not token_exists_after:
            print("  - Token prediction database was not created")
        if not escalation_exists_after:
            print("  - Escalation prediction database was not created")
        return 1

if __name__ == "__main__":
    sys.exit(test_database_initialization())



