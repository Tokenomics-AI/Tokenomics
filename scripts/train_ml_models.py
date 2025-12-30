#!/usr/bin/env python3
"""Train all ML models when enough data is available."""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.training_config import TRAINING_CONFIG


def train_models():
    """Train all ML models."""
    print("=" * 80)
    print("ML Models Training")
    print("=" * 80)
    print()
    
    config = TRAINING_CONFIG
    output_dir = Path(config.get("output_dir", "training_data"))
    output_dir.mkdir(exist_ok=True)
    
    # Initialize platform
    print("1. Initializing TokenomicsPlatform")
    print("-" * 80)
    try:
        from tokenomics.core import TokenomicsPlatform
        from tokenomics.config import TokenomicsConfig
        
        platform_config = TokenomicsConfig.from_env()
        platform = TokenomicsPlatform(config=platform_config)
        print("✓ Platform initialized")
    except Exception as e:
        print(f"✗ ERROR: Failed to initialize platform: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # Check data availability
    print("2. Checking data availability")
    print("-" * 80)
    
    min_samples = config["min_samples"]
    results = {
        "token_predictor": {"trained": False},
        "escalation_predictor": {"trained": False},
        "complexity_classifier": {"trained": False},
    }
    
    # Get current stats
    if platform.token_predictor and platform.token_predictor.data_collector:
        stats = platform.token_predictor.data_collector.get_stats()
        token_count = stats.get("token_prediction", {}).get("total_samples", 0)
        escalation_count = stats.get("escalation_prediction", {}).get("total_samples", 0)
        complexity_count = stats.get("complexity_prediction", {}).get("total_samples", 0)
        
        print(f"   Token predictions: {token_count}/{min_samples['token_predictor']}")
        print(f"   Escalation predictions: {escalation_count}/{min_samples['escalation_predictor']}")
        print(f"   Complexity predictions: {complexity_count}/{min_samples['complexity_classifier']}")
        print()
        
        # Train token predictor
        print("3. Training Token Predictor")
        print("-" * 80)
        if token_count >= min_samples["token_predictor"]:
            try:
                success = platform.train_token_predictor(min_samples=min_samples["token_predictor"])
                if success:
                    results["token_predictor"] = {
                        "trained": True,
                        "samples_used": token_count,
                        "model_type": "XGBoost Regressor",
                        "trained_at": datetime.now().isoformat(),
                    }
                    print("✓ Token Predictor trained successfully!")
                else:
                    print("✗ Token Predictor training failed")
                    results["token_predictor"]["error"] = "Training returned False"
            except Exception as e:
                print(f"✗ Token Predictor training error: {e}")
                results["token_predictor"]["error"] = str(e)
        else:
            print(f"⚠ Not enough data: {token_count}/{min_samples['token_predictor']} samples")
            results["token_predictor"]["error"] = f"Insufficient data: {token_count}/{min_samples['token_predictor']}"
        
        print()
        
        # Train escalation predictor
        print("4. Training Escalation Predictor")
        print("-" * 80)
        if escalation_count >= min_samples["escalation_predictor"]:
            try:
                success = platform.train_escalation_predictor(min_samples=min_samples["escalation_predictor"])
                if success:
                    results["escalation_predictor"] = {
                        "trained": True,
                        "samples_used": escalation_count,
                        "model_type": "XGBoost Classifier",
                        "trained_at": datetime.now().isoformat(),
                    }
                    print("✓ Escalation Predictor trained successfully!")
                else:
                    print("✗ Escalation Predictor training failed")
                    results["escalation_predictor"]["error"] = "Training returned False"
            except Exception as e:
                print(f"✗ Escalation Predictor training error: {e}")
                results["escalation_predictor"]["error"] = str(e)
        else:
            print(f"⚠ Not enough data: {escalation_count}/{min_samples['escalation_predictor']} samples")
            results["escalation_predictor"]["error"] = f"Insufficient data: {escalation_count}/{min_samples['escalation_predictor']}"
        
        print()
        
        # Train complexity classifier
        print("5. Training Complexity Classifier")
        print("-" * 80)
        if complexity_count >= min_samples["complexity_classifier"]:
            try:
                success = platform.train_complexity_classifier(min_samples=min_samples["complexity_classifier"])
                if success:
                    results["complexity_classifier"] = {
                        "trained": True,
                        "samples_used": complexity_count,
                        "model_type": "XGBoost Multi-Class Classifier",
                        "trained_at": datetime.now().isoformat(),
                    }
                    print("✓ Complexity Classifier trained successfully!")
                else:
                    print("✗ Complexity Classifier training failed")
                    results["complexity_classifier"]["error"] = "Training returned False"
            except Exception as e:
                print(f"✗ Complexity Classifier training error: {e}")
                results["complexity_classifier"]["error"] = str(e)
        else:
            print(f"⚠ Not enough data: {complexity_count}/{min_samples['complexity_classifier']} samples")
            results["complexity_classifier"]["error"] = f"Insufficient data: {complexity_count}/{min_samples['complexity_classifier']}"
        
        print()
        
        # Save results
        print("6. Saving training results")
        print("-" * 80)
        
        results_file = output_dir / "training_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved training results: {results_file}")
        
        # Summary
        print()
        print("=" * 80)
        print("Training Summary")
        print("=" * 80)
        
        trained_count = sum(1 for r in results.values() if r.get("trained", False))
        total_models = len(results)
        
        for model_name, result in results.items():
            status = "✓ TRAINED" if result.get("trained") else "✗ NOT TRAINED"
            print(f"   {model_name}: {status}")
            if result.get("error"):
                print(f"      Error: {result['error']}")
        
        print()
        if trained_count == total_models:
            print("✓ All models trained successfully!")
            print("   Next step: python3 scripts/evaluate_ml_models.py")
        else:
            print(f"⚠ {trained_count}/{total_models} models trained")
            print("   Some models need more data. Run data collection again:")
            print("   python3 scripts/collect_training_data.py")
        
        return 0 if trained_count == total_models else 1
    
    else:
        print("✗ ERROR: Platform ML components not available")
        return 1


if __name__ == "__main__":
    sys.exit(train_models())



