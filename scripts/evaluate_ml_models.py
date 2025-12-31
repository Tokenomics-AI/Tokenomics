#!/usr/bin/env python3
"""Evaluate trained ML models on test set."""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.training_config import TRAINING_CONFIG


def load_test_queries() -> List[Dict]:
    """Load test queries from JSON file."""
    output_dir = Path(TRAINING_CONFIG.get("output_dir", "training_data"))
    test_file = output_dir / "test_queries.json"
    
    if not test_file.exists():
        print(f"✗ ERROR: Test queries file not found: {test_file}")
        print("   Run data preparation script first:")
        print("   python3 scripts/prepare_customer_support_dataset.py")
        return []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get("queries", [])


def calculate_token_metrics(predictions: List[float], actuals: List[int]) -> Dict:
    """Calculate token prediction metrics."""
    import numpy as np
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Remove zeros/invalid values
    valid_mask = (predictions > 0) & (actuals > 0)
    if valid_mask.sum() == 0:
        return {"error": "No valid predictions"}
    
    pred_valid = predictions[valid_mask]
    actual_valid = actuals[valid_mask]
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred_valid - actual_valid))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((pred_valid - actual_valid) ** 2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((pred_valid - actual_valid) / actual_valid)) * 100
    
    # Accuracy within 20% buffer
    within_20_pct = np.sum(np.abs(pred_valid - actual_valid) / actual_valid <= 0.2) / len(pred_valid) * 100
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "accuracy_within_20pct": float(within_20_pct),
        "num_samples": int(valid_mask.sum()),
    }


def calculate_classification_metrics(predictions: List[str], actuals: List[str]) -> Dict:
    """Calculate classification metrics (for complexity and escalation)."""
    from collections import Counter
    
    if len(predictions) != len(actuals):
        return {"error": "Mismatched prediction and actual lengths"}
    
    # Accuracy
    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    accuracy = correct / len(predictions) * 100 if predictions else 0
    
    # Per-class accuracy
    classes = set(predictions + actuals)
    per_class_accuracy = {}
    for cls in classes:
        cls_predictions = [p for p, a in zip(predictions, actuals) if a == cls]
        cls_correct = sum(1 for p in cls_predictions if p == cls)
        per_class_accuracy[cls] = (cls_correct / len(cls_predictions) * 100) if cls_predictions else 0
    
    # Confusion matrix
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    for p, a in zip(predictions, actuals):
        confusion_matrix[a][p] += 1
    
    return {
        "accuracy": float(accuracy),
        "per_class_accuracy": {k: float(v) for k, v in per_class_accuracy.items()},
        "confusion_matrix": {k: dict(v) for k, v in confusion_matrix.items()},
        "num_samples": len(predictions),
    }


def calculate_escalation_metrics(predicted_probs: List[float], actual_escalated: List[bool], threshold: float = 0.5) -> Dict:
    """Calculate escalation prediction metrics."""
    predicted_escalated = [p >= threshold for p in predicted_probs]
    
    if len(predicted_escalated) != len(actual_escalated):
        return {"error": "Mismatched prediction and actual lengths"}
    
    # Confusion matrix
    tp = sum(1 for p, a in zip(predicted_escalated, actual_escalated) if p and a)
    fp = sum(1 for p, a in zip(predicted_escalated, actual_escalated) if p and not a)
    tn = sum(1 for p, a in zip(predicted_escalated, actual_escalated) if not p and not a)
    fn = sum(1 for p, a in zip(predicted_escalated, actual_escalated) if not p and a)
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        "accuracy": float(accuracy * 100),
        "precision": float(precision * 100),
        "recall": float(recall * 100),
        "f1_score": float(f1 * 100),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "num_samples": len(predicted_escalated),
    }


def load_evaluation_checkpoint(output_dir: Path) -> Optional[Dict]:
    """Load evaluation checkpoint if exists."""
    checkpoint_file = output_dir / "evaluation_checkpoint.json"
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data
        except (json.JSONDecodeError, IOError) as e:
            print(f"   ⚠ WARNING: Failed to load checkpoint: {e}. Starting fresh.")
            checkpoint_file.unlink(missing_ok=True)
    return None


def save_evaluation_checkpoint(
    output_dir: Path,
    processed_count: int,
    token_predictions: List[float],
    token_actuals: List[int],
    escalation_predictions: List[float],
    escalation_actuals: List[bool],
    complexity_predictions: List[str],
    complexity_actuals: List[str],
    successful_count: int,
    failed_count: int,
):
    """Save evaluation checkpoint."""
    checkpoint_file = output_dir / "evaluation_checkpoint.json"
    checkpoint_data = {
        "processed_count": processed_count,
        "token_predictions": token_predictions,
        "token_actuals": token_actuals,
        "escalation_predictions": escalation_predictions,
        "escalation_actuals": escalation_actuals,
        "complexity_predictions": complexity_predictions,
        "complexity_actuals": complexity_actuals,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"   ✓ Checkpoint saved at query {processed_count}")
    except Exception as e:
        print(f"   ⚠ WARNING: Failed to save checkpoint: {e}")


def evaluate_models(max_queries: int = None):
    """Evaluate trained models on test set."""
    print("=" * 80)
    print("ML Models Evaluation")
    print("=" * 80)
    print()
    
    config = TRAINING_CONFIG
    output_dir = Path(config.get("output_dir", "training_data"))
    output_dir.mkdir(exist_ok=True)
    
    # Load test queries
    print("1. Loading test queries")
    print("-" * 80)
    test_queries = load_test_queries()
    
    if not test_queries:
        return 1
    
    if max_queries:
        test_queries = test_queries[:max_queries]
        print(f"✓ Limited to {max_queries} queries for evaluation")
    else:
        print(f"✓ Loaded {len(test_queries)} test queries")
    
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
        
        # Check if models are trained
        token_trained = platform.token_predictor and platform.token_predictor.model_trained if platform.token_predictor else False
        escalation_trained = platform.escalation_predictor and platform.escalation_predictor.model_trained if platform.escalation_predictor else False
        complexity_trained = platform.complexity_classifier and platform.complexity_classifier.model_trained if platform.complexity_classifier else False
        
        print(f"   Token Predictor: {'✓ Trained' if token_trained else '✗ Using Heuristic'}")
        print(f"   Escalation Predictor: {'✓ Trained' if escalation_trained else '✗ Using Heuristic'}")
        print(f"   Complexity Classifier: {'✓ Trained' if complexity_trained else '✗ Using Heuristic'}")
    except Exception as e:
        print(f"✗ ERROR: Failed to initialize platform: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # Check for existing checkpoint
    print("3. Checking for existing checkpoint")
    print("-" * 80)
    checkpoint = load_evaluation_checkpoint(output_dir)
    
    if checkpoint:
        start_index = checkpoint.get("processed_count", 0)
        token_predictions = checkpoint.get("token_predictions", [])
        token_actuals = checkpoint.get("token_actuals", [])
        escalation_predictions = checkpoint.get("escalation_predictions", [])
        escalation_actuals = checkpoint.get("escalation_actuals", [])
        complexity_predictions = checkpoint.get("complexity_predictions", [])
        complexity_actuals = checkpoint.get("complexity_actuals", [])
        successful = checkpoint.get("successful_count", 0)
        failed = checkpoint.get("failed_count", 0)
        
        print(f"✓ Found checkpoint: {start_index} queries already processed")
        print(f"   Resuming from query {start_index + 1}/{len(test_queries)}")
        
        # Skip already processed queries
        test_queries = test_queries[start_index:]
    else:
        start_index = 0
        token_predictions = []
        token_actuals = []
        escalation_predictions = []
        escalation_actuals = []
        complexity_predictions = []
        complexity_actuals = []
        successful = 0
        failed = 0
        print("✓ No checkpoint found. Starting fresh evaluation.")
    
    print()
    
    # Run evaluation
    print("4. Running evaluation on test queries")
    print("-" * 80)
    
    for idx, query_data in enumerate(test_queries, start=1):
        actual_idx = start_index + idx
        query_text = query_data["query"]
        
        try:
            # Run query through platform
            result = platform.query(
                query=query_text,
                use_cache=False,  # Disable cache for fresh evaluation
                use_bandit=True,
                use_compression=True,
            )
            
            # Extract token prediction
            if "predicted_max_tokens" in result and "output_tokens" in result:
                token_predictions.append(result["predicted_max_tokens"])
                token_actuals.append(result["output_tokens"])
            
            # Extract complexity prediction (from plan or result)
            if "plan" in result and result["plan"]:
                predicted_complexity = result["plan"].complexity.value if hasattr(result["plan"], "complexity") else None
                # Get actual complexity from orchestrator heuristic
                actual_complexity = platform.orchestrator._analyze_complexity_heuristic(query_text).value
                if predicted_complexity and actual_complexity:
                    complexity_predictions.append(predicted_complexity)
                    complexity_actuals.append(actual_complexity)
            
            # Extract escalation prediction
            # Check if cascading was used and get escalation likelihood
            # Note: Escalation prediction probability is not directly in result
            # We can check cascading metrics or predict separately if needed
            if platform.escalation_predictor and platform.complexity_classifier:
                try:
                    # Get query embedding for prediction
                    query_embedding = None
                    if platform.memory and platform.memory.use_semantic_cache and platform.memory.embedding_model:
                        try:
                            query_embedding = platform.memory.get_embedding(query_text)
                        except Exception:
                            pass
                    
                    # Get complexity for escalation prediction
                    complexity = result.get("query_type") or "medium"
                    context_quality = result.get("plan", {}).context_quality_score if hasattr(result.get("plan", {}), "context_quality_score") else 0.8
                    query_tokens = result.get("input_tokens", 0)
                    
                    # Predict escalation likelihood
                    escalation_prob = platform.escalation_predictor.predict(
                        query=query_text,
                        complexity=complexity,
                        context_quality_score=context_quality,
                        query_tokens=query_tokens,
                        query_embedding=query_embedding,
                    )
                    
                    # Check if escalation actually happened (from cascading metrics or result)
                    # This is approximate - actual escalation requires cascading to be enabled
                    escalation_predictions.append(escalation_prob)
                    # For actual escalation, we'd need to check cascading outcomes
                    # For now, use a placeholder
                    escalation_actuals.append(False)  # Would need actual cascading data
                except Exception as e:
                    pass  # Skip if escalation prediction fails
            
            successful += 1
            
            # Add delay between queries to avoid rate limiting (1.5 seconds)
            time.sleep(1.5)
            
            # Save checkpoint every 10 queries
            if actual_idx % 10 == 0:
                print(f"   Processed {actual_idx}/{len(test_queries) + start_index} queries...")
                save_evaluation_checkpoint(
                    output_dir,
                    actual_idx,
                    token_predictions,
                    token_actuals,
                    escalation_predictions,
                    escalation_actuals,
                    complexity_predictions,
                    complexity_actuals,
                    successful,
                    failed,
                )
        
        except Exception as e:
            failed += 1
            print(f"   ✗ Query {actual_idx} failed: {str(e)[:100]}")
            # Add delay even on failure to avoid rate limiting
            time.sleep(1.5)
            # Save checkpoint even on failure
            if actual_idx % 10 == 0:
                save_evaluation_checkpoint(
                    output_dir,
                    actual_idx,
                    token_predictions,
                    token_actuals,
                    escalation_predictions,
                    escalation_actuals,
                    complexity_predictions,
                    complexity_actuals,
                    successful,
                    failed,
                )
            continue
    
    print()
    print(f"✓ Evaluation complete: {successful} successful, {failed} failed")
    print()
    
    # Remove checkpoint file after successful completion
    checkpoint_file = output_dir / "evaluation_checkpoint.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("✓ Evaluation checkpoint removed after successful completion.")
    print()
    
    # Calculate metrics
    print("5. Calculating evaluation metrics")
    print("-" * 80)
    
    evaluation_results = {
        "token_predictor": {},
        "escalation_predictor": {},
        "complexity_classifier": {},
        "summary": {},
    }
    
    # Token Predictor metrics
    if token_predictions and token_actuals:
        token_metrics = calculate_token_metrics(token_predictions, token_actuals)
        evaluation_results["token_predictor"] = token_metrics
        print("✓ Token Predictor metrics calculated")
        if "error" not in token_metrics:
            print(f"   MAE: {token_metrics['mae']:.2f}")
            print(f"   RMSE: {token_metrics['rmse']:.2f}")
            print(f"   MAPE: {token_metrics['mape']:.2f}%")
            print(f"   Accuracy (within 20%): {token_metrics['accuracy_within_20pct']:.2f}%")
    else:
        print("⚠ No token prediction data collected")
        evaluation_results["token_predictor"] = {"error": "No data collected"}
    
    print()
    
    # Complexity Classifier metrics
    if complexity_predictions and complexity_actuals:
        complexity_metrics = calculate_classification_metrics(complexity_predictions, complexity_actuals)
        evaluation_results["complexity_classifier"] = complexity_metrics
        print("✓ Complexity Classifier metrics calculated")
        if "error" not in complexity_metrics:
            print(f"   Accuracy: {complexity_metrics['accuracy']:.2f}%")
            for cls, acc in complexity_metrics['per_class_accuracy'].items():
                print(f"   {cls.capitalize()} accuracy: {acc:.2f}%")
    else:
        print("⚠ No complexity prediction data collected")
        evaluation_results["complexity_classifier"] = {"error": "No data collected"}
    
    print()
    
    # Escalation Predictor metrics
    # Note: Escalation prediction requires checking cascading outcomes
    # This is more complex and may need separate evaluation
    print("⚠ Escalation Predictor evaluation requires cascading data")
    print("   (Can be evaluated separately by checking cascading outcomes)")
    evaluation_results["escalation_predictor"] = {"note": "Requires cascading outcome analysis"}
    
    print()
    
    # Summary
    evaluation_results["summary"] = {
        "total_queries": len(test_queries),
        "successful": successful,
        "failed": failed,
        "models_evaluated": {
            "token_predictor": "error" not in evaluation_results["token_predictor"],
            "complexity_classifier": "error" not in evaluation_results["complexity_classifier"],
            "escalation_predictor": False,  # Not fully evaluated
        }
    }
    
    # Save results
    print("6. Saving evaluation results")
    print("-" * 80)
    
    results_file_json = output_dir / "evaluation_report.json"
    with open(results_file_json, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"✓ Saved JSON report: {results_file_json}")
    
    # Generate markdown report
    results_file_md = output_dir / "evaluation_report.md"
    with open(results_file_md, 'w', encoding='utf-8') as f:
        f.write("# ML Models Evaluation Report\n\n")
        f.write(f"**Evaluation Date**: {Path(__file__).stat().st_mtime}\n\n")
        f.write(f"**Test Queries**: {len(test_queries)}\n")
        f.write(f"**Successful**: {successful}\n")
        f.write(f"**Failed**: {failed}\n\n")
        
        f.write("## Token Predictor\n\n")
        if "error" not in evaluation_results["token_predictor"]:
            metrics = evaluation_results["token_predictor"]
            f.write(f"- **MAE**: {metrics['mae']:.2f} tokens\n")
            f.write(f"- **RMSE**: {metrics['rmse']:.2f} tokens\n")
            f.write(f"- **MAPE**: {metrics['mape']:.2f}%\n")
            f.write(f"- **Accuracy (within 20%)**: {metrics['accuracy_within_20pct']:.2f}%\n")
            f.write(f"- **Samples**: {metrics['num_samples']}\n")
        else:
            f.write(f"- Error: {evaluation_results['token_predictor']['error']}\n")
        
        f.write("\n## Complexity Classifier\n\n")
        if "error" not in evaluation_results["complexity_classifier"]:
            metrics = evaluation_results["complexity_classifier"]
            f.write(f"- **Overall Accuracy**: {metrics['accuracy']:.2f}%\n")
            f.write(f"- **Per-Class Accuracy**:\n")
            for cls, acc in metrics['per_class_accuracy'].items():
                f.write(f"  - {cls.capitalize()}: {acc:.2f}%\n")
            f.write(f"- **Samples**: {metrics['num_samples']}\n")
        else:
            f.write(f"- Error: {evaluation_results['complexity_classifier']['error']}\n")
        
        f.write("\n## Escalation Predictor\n\n")
        f.write("- Evaluation requires cascading outcome analysis\n")
    
    print(f"✓ Saved Markdown report: {results_file_md}")
    
    print()
    print("=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"✓ Reports saved to: {output_dir}")
    print(f"   - evaluation_report.json")
    print(f"   - evaluation_report.md")
    
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ML models on test set")
    parser.add_argument("--max-queries", type=int, help="Maximum number of test queries to evaluate")
    args = parser.parse_args()
    
    sys.exit(evaluate_models(max_queries=args.max_queries))




