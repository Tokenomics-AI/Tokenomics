#!/usr/bin/env python3
"""Monitor data collection and automatically train models when ready."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenomics.ml.unified_data_collector import UnifiedDataCollector
from scripts.training_config import TRAINING_CONFIG


def check_and_train():
    """Check data availability and train models when ready."""
    dc = UnifiedDataCollector()
    stats = dc.get_stats()
    
    t = stats.get('token_prediction', {}).get('total_samples', 0)
    e = stats.get('escalation_prediction', {}).get('total_samples', 0)
    c = stats.get('complexity_prediction', {}).get('total_samples', 0)
    
    min_samples = TRAINING_CONFIG["min_samples"]
    
    print(f"Current samples: Token={t}, Escalation={e}, Complexity={c}")
    
    token_ready = t >= min_samples["token_predictor"]
    escalation_ready = e >= min_samples["escalation_predictor"]
    complexity_ready = c >= min_samples["complexity_classifier"]
    
    if token_ready and escalation_ready and complexity_ready:
        print("✓ All models ready for training!")
        print("Running training script...")
        import subprocess
        result = subprocess.run(
            ["python3", str(Path(__file__).parent / "train_ml_models.py")],
            cwd=Path(__file__).parent.parent
        )
        return result.returncode == 0
    else:
        print("⏳ Waiting for more data...")
        if not token_ready:
            print(f"   Token Predictor: need {min_samples['token_predictor'] - t} more")
        if not escalation_ready:
            print(f"   Escalation Predictor: need {min_samples['escalation_predictor'] - e} more")
        if not complexity_ready:
            print(f"   Complexity Classifier: need {min_samples['complexity_classifier'] - c} more")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor data collection and train when ready")
    parser.add_argument("--check-interval", type=int, default=60, help="Check interval in seconds (default: 60)")
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    args = parser.parse_args()
    
    if args.once:
        sys.exit(0 if check_and_train() else 1)
    else:
        print(f"Monitoring data collection (checking every {args.check_interval} seconds)...")
        print("Press Ctrl+C to stop")
        try:
            while True:
                if check_and_train():
                    print("Training completed! Exiting monitor.")
                    break
                time.sleep(args.check_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")


