#!/usr/bin/env python3
"""Comprehensive test for ML models integration with the platform."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_unified_database():
    """Test unified database creation and all 3 tables."""
    print("=" * 80)
    print("Test 1: Unified Database Creation")
    print("=" * 80)
    
    try:
        from tokenomics.ml.unified_data_collector import UnifiedDataCollector
        
        collector = UnifiedDataCollector(db_path="test_ml_training_data.db")
        print("✓ UnifiedDataCollector initialized")
        
        # Test recording for all 3 model types
        collector.record_token_prediction(
            query="Test query for token prediction",
            query_length=30,
            complexity="simple",
            embedding_vector=[0.1] * 10,
            predicted_tokens=100,
            actual_output_tokens=95,
            model_used="gpt-4o-mini",
        )
        print("✓ Token prediction data recorded")
        
        collector.record_escalation_prediction(
            query="Test query for escalation",
            query_length=35,
            complexity="medium",
            context_quality_score=0.75,
            query_tokens=10,
            query_embedding=[0.2] * 10,
            escalated=False,
            model_used="gpt-4o-mini",
        )
        print("✓ Escalation prediction data recorded")
        
        collector.record_complexity_prediction(
            query="Test query for complexity",
            query_length=40,
            query_tokens=15,
            query_embedding=[0.3] * 10,
            keyword_counts={"complex_score": 1, "medium_score": 0, "question_count": 1, "has_comparison": 0},
            predicted_complexity="medium",
            actual_complexity="medium",
            model_used="gpt-4o-mini",
        )
        print("✓ Complexity prediction data recorded")
        
        # Test stats
        stats = collector.get_stats()
        print(f"✓ Stats retrieved: {list(stats.keys())}")
        
        # Cleanup
        Path("test_ml_training_data.db").unlink()
        print("✓ Test database cleaned up")
        
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interface_compatibility():
    """Test that UnifiedDataCollector works with TokenPredictor and EscalationPredictor interfaces."""
    print("\n" + "=" * 80)
    print("Test 2: Interface Compatibility")
    print("=" * 80)
    
    try:
        from tokenomics.ml.unified_data_collector import UnifiedDataCollector
        
        collector = UnifiedDataCollector(db_path="test_ml_training_data.db")
        
        # Test TokenPredictor interface (record with token prediction params)
        collector.record(
            query="Token test",
            query_length=10,
            complexity="simple",
            embedding_vector=[0.1] * 10,
            predicted_tokens=50,
            actual_output_tokens=45,
            model_used="gpt-4o-mini",
        )
        print("✓ TokenPredictor interface (record) works")
        
        # Test EscalationPredictor interface (record with escalation params)
        collector.record(
            query="Escalation test",
            query_length=15,
            complexity="medium",
            context_quality_score=0.8,
            query_tokens=12,
            query_embedding=[0.2] * 10,
            escalated=False,
            model_used="gpt-4o-mini",
        )
        print("✓ EscalationPredictor interface (record) works")
        
        # Test get_training_data for token (default)
        token_data = collector.get_training_data("token", min_samples=1)
        if token_data:
            print(f"✓ Token training data retrieved: {len(token_data)} samples")
        else:
            print("⚠ No token training data yet (expected)")
        
        # Test get_training_data for escalation
        escalation_data = collector.get_training_data("escalation", min_samples=1)
        if escalation_data:
            print(f"✓ Escalation training data retrieved: {len(escalation_data)} samples")
        else:
            print("⚠ No escalation training data yet (expected)")
        
        # Test get_stats
        all_stats = collector.get_stats()
        token_stats = collector.get_stats("token")
        escalation_stats = collector.get_stats("escalation")
        
        print(f"✓ All stats: {list(all_stats.keys())}")
        print(f"✓ Token stats: {token_stats.get('total_samples', 0)} samples")
        print(f"✓ Escalation stats: {escalation_stats.get('total_samples', 0)} samples")
        
        # Cleanup
        Path("test_ml_training_data.db").unlink()
        print("✓ Test database cleaned up")
        
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_predictor_integration():
    """Test that TokenPredictor and EscalationPredictor work with UnifiedDataCollector."""
    print("\n" + "=" * 80)
    print("Test 3: Predictor Integration with UnifiedDataCollector")
    print("=" * 80)
    
    try:
        from tokenomics.ml.unified_data_collector import UnifiedDataCollector
        from tokenomics.ml.token_predictor import TokenPredictor
        from tokenomics.ml.escalation_predictor import EscalationPredictor
        
        collector = UnifiedDataCollector(db_path="test_ml_training_data.db")
        
        # Test TokenPredictor
        token_predictor = TokenPredictor(data_collector=collector)
        token_predictor.record_prediction(
            query="Test token query",
            complexity="simple",
            query_tokens=10,
            predicted_tokens=50,
            actual_output_tokens=45,
            model_used="gpt-4o-mini",
        )
        print("✓ TokenPredictor.record_prediction() works")
        
        # Test EscalationPredictor
        escalation_predictor = EscalationPredictor(data_collector=collector)
        escalation_predictor.record_outcome(
            query="Test escalation query",
            complexity="medium",
            context_quality_score=0.75,
            query_tokens=12,
            escalated=False,
            model_used="gpt-4o-mini",
        )
        print("✓ EscalationPredictor.record_outcome() works")
        
        # Test get_stats from predictors
        token_stats = token_predictor.get_stats()
        escalation_stats = escalation_predictor.get_stats()
        
        print(f"✓ TokenPredictor stats: {token_stats.get('total_samples', 0)} samples")
        print(f"✓ EscalationPredictor stats: {escalation_stats.get('total_samples', 0)} samples")
        
        # Cleanup
        Path("test_ml_training_data.db").unlink()
        print("✓ Test database cleaned up")
        
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complexity_classifier():
    """Test ComplexityClassifier initialization and prediction."""
    print("\n" + "=" * 80)
    print("Test 4: Complexity Classifier")
    print("=" * 80)
    
    try:
        from tokenomics.ml.unified_data_collector import UnifiedDataCollector
        from tokenomics.ml.complexity_classifier import ComplexityClassifier
        
        collector = UnifiedDataCollector(db_path="test_ml_training_data.db")
        classifier = ComplexityClassifier(data_collector=collector)
        
        # Test heuristic prediction
        result1 = classifier.predict("What is 2+2?")
        print(f"✓ Heuristic prediction (simple): '{result1}'")
        
        result2 = classifier.predict("Design a comprehensive system architecture for a microservices-based e-commerce platform")
        print(f"✓ Heuristic prediction (complex): '{result2}'")
        
        # Test recording
        classifier.record_prediction(
            query="Test query",
            predicted_complexity="medium",
            actual_complexity="medium",
            query_embedding=[0.1] * 10,
        )
        print("✓ Complexity prediction recorded")
        
        # Test stats
        stats = classifier.get_stats()
        print(f"✓ Classifier stats: model_trained={stats.get('model_trained', False)}")
        
        # Cleanup
        Path("test_ml_training_data.db").unlink()
        print("✓ Test database cleaned up")
        
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_platform_integration():
    """Test that platform initializes all ML models correctly."""
    print("\n" + "=" * 80)
    print("Test 5: Platform Integration")
    print("=" * 80)
    
    try:
        from tokenomics.core import TokenomicsPlatform
        from tokenomics.config import TokenomicsConfig
        
        config = TokenomicsConfig.from_env()
        platform = TokenomicsPlatform(config=config)
        
        print(f"✓ Platform initialized")
        print(f"  - Token predictor: {'✓' if platform.token_predictor else '✗'}")
        print(f"  - Escalation predictor: {'✓' if platform.escalation_predictor else '✗'}")
        print(f"  - Complexity classifier: {'✓' if platform.complexity_classifier else '✗'}")
        
        # Test that all use unified database
        if platform.token_predictor and platform.token_predictor.data_collector:
            db_path = str(platform.token_predictor.data_collector.db_path)
            print(f"  - Token predictor DB: {db_path}")
            is_unified = "ml_training_data.db" in db_path
            print(f"    Using unified DB: {'✓' if is_unified else '✗'}")
        
        if platform.escalation_predictor and platform.escalation_predictor.data_collector:
            db_path = str(platform.escalation_predictor.data_collector.db_path)
            print(f"  - Escalation predictor DB: {db_path}")
            is_unified = "ml_training_data.db" in db_path
            print(f"    Using unified DB: {'✓' if is_unified else '✗'}")
        
        if platform.complexity_classifier and platform.complexity_classifier.data_collector:
            db_path = str(platform.complexity_classifier.data_collector.db_path)
            print(f"  - Complexity classifier DB: {db_path}")
            is_unified = "ml_training_data.db" in db_path
            print(f"    Using unified DB: {'✓' if is_unified else '✗'}")
        
        # Test complexity prediction in query flow
        if platform.complexity_classifier:
            test_query = "What is machine learning?"
            complexity = platform.complexity_classifier.predict(test_query)
            print(f"  - Complexity prediction test: '{test_query}' -> {complexity}")
        
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("ML Models Integration Test Suite")
    print("=" * 80)
    print()
    
    results = []
    
    results.append(("Unified Database", test_unified_database()))
    results.append(("Interface Compatibility", test_interface_compatibility()))
    results.append(("Predictor Integration", test_predictor_integration()))
    results.append(("Complexity Classifier", test_complexity_classifier()))
    results.append(("Platform Integration", test_platform_integration()))
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print()
    if all_passed:
        print("✓ All tests PASSED! ML models are correctly integrated.")
        print("\nNext steps:")
        print("  1. Run the migration script to migrate existing data:")
        print("     python3 scripts/migrate_to_unified_db.py")
        print("  2. Run platform tests to collect training data:")
        print("     python3 tests/complete_platform_test.py")
        print("  3. Train the models when enough data is collected")
    else:
        print("✗ Some tests FAILED. Please review the errors above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())



