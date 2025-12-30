#!/usr/bin/env python3
"""Generate comprehensive training and performance documentation."""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.training_config import TRAINING_CONFIG


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load JSON file, return empty dict if not found."""
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with specified decimals."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def generate_training_summary(output_dir: Path) -> str:
    """Generate training summary report."""
    dataset_stats = load_json_file(output_dir / "dataset_stats.json")
    training_results = load_json_file(output_dir / "training_results.json")
    
    md = []
    md.append("# ML Models Training Summary")
    md.append("")
    md.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    
    # Dataset Statistics
    md.append("## Dataset Statistics")
    md.append("")
    if dataset_stats:
        cleaning_stats = dataset_stats.get("cleaning_stats", {})
        md.append(f"- **Total Original Rows**: {dataset_stats.get('total_original', 'N/A'):,}")
        md.append(f"- **After Cleaning**: {dataset_stats.get('after_cleaning', 'N/A'):,}")
        md.append(f"- **Training Queries**: {dataset_stats.get('training_count', 'N/A'):,}")
        md.append(f"- **Test Queries**: {dataset_stats.get('test_count', 'N/A'):,}")
        md.append("")
        md.append("### Cleaning Statistics")
        md.append(f"- Empty/Too Short: {cleaning_stats.get('too_short', 0)}")
        md.append(f"- Too Long: {cleaning_stats.get('too_long', 0)}")
        md.append(f"- Duplicates Removed: {cleaning_stats.get('duplicates', 0)}")
        
        if "query_length_stats" in dataset_stats:
            length_stats = dataset_stats["query_length_stats"]
            md.append("")
            md.append("### Query Length Statistics")
            md.append(f"- Minimum: {length_stats.get('min', 'N/A')} characters")
            md.append(f"- Maximum: {length_stats.get('max', 'N/A')} characters")
            md.append(f"- Average: {format_number(length_stats.get('avg', 0))} characters")
    else:
        md.append("*Dataset statistics not available*")
    md.append("")
    
    # Data Collection Summary
    md.append("## Data Collection Summary")
    md.append("")
    checkpoint = load_json_file(output_dir / "checkpoint.json")
    if checkpoint:
        data_stats = checkpoint.get("data_stats", {})
        md.append(f"- **Queries Processed**: {checkpoint.get('queries_run', 0):,}")
        md.append(f"- **Successful**: {checkpoint.get('successful', 0):,}")
        md.append(f"- **Failed**: {checkpoint.get('failed', 0):,}")
        md.append("")
        md.append("### Samples Collected")
        md.append(f"- **Token Predictions**: {data_stats.get('token_predictions', 0):,}")
        md.append(f"- **Escalation Predictions**: {data_stats.get('escalation_predictions', 0):,}")
        md.append(f"- **Complexity Predictions**: {data_stats.get('complexity_predictions', 0):,}")
        if checkpoint.get("timestamp"):
            md.append(f"- **Last Updated**: {checkpoint['timestamp']}")
    else:
        md.append("*Checkpoint data not available*")
    md.append("")
    
    # Training Status
    md.append("## Model Training Status")
    md.append("")
    if training_results:
        min_samples = TRAINING_CONFIG["min_samples"]
        
        for model_name, result in training_results.items():
            model_display = model_name.replace("_", " ").title()
            md.append(f"### {model_display}")
            md.append("")
            
            if result.get("trained"):
                md.append(f"- **Status**: ✓ Trained Successfully")
                md.append(f"- **Samples Used**: {result.get('samples_used', 'N/A'):,}")
                md.append(f"- **Model Type**: {result.get('model_type', 'N/A')}")
                md.append(f"- **Trained At**: {result.get('trained_at', 'N/A')}")
                md.append(f"- **Minimum Required**: {min_samples.get(model_name, 'N/A'):,}")
            else:
                md.append(f"- **Status**: ✗ Not Trained")
                if "error" in result:
                    md.append(f"- **Error**: {result['error']}")
                md.append(f"- **Minimum Required**: {min_samples.get(model_name, 'N/A'):,}")
            md.append("")
    else:
        md.append("*Training results not available*")
        md.append("")
        md.append("Run `python3 scripts/train_ml_models.py` to train the models.")
    md.append("")
    
    # Model Specifications
    md.append("## Model Specifications")
    md.append("")
    md.append("### Token Predictor")
    md.append("- **Type**: XGBoost Regressor")
    md.append("- **Purpose**: Predict output token count for queries")
    md.append("- **Features**: Query length, complexity, embedding vector, context quality")
    md.append("- **Minimum Samples**: 500")
    md.append("")
    md.append("### Escalation Predictor")
    md.append("- **Type**: XGBoost Classifier")
    md.append("- **Purpose**: Predict likelihood of needing model escalation")
    md.append("- **Features**: Query complexity, context quality score, query tokens, embedding")
    md.append("- **Minimum Samples**: 100")
    md.append("")
    md.append("### Complexity Classifier")
    md.append("- **Type**: XGBoost Multi-Class Classifier")
    md.append("- **Purpose**: Classify query complexity (simple/medium/complex)")
    md.append("- **Features**: Query length, token count, embedding, keyword counts")
    md.append("- **Minimum Samples**: 100")
    md.append("")
    
    return "\n".join(md)


def generate_performance_report(output_dir: Path) -> str:
    """Generate model performance report."""
    evaluation_results = load_json_file(output_dir / "evaluation_report.json")
    
    md = []
    md.append("# ML Models Performance Report")
    md.append("")
    md.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    
    if not evaluation_results:
        md.append("*Evaluation results not available*")
        md.append("")
        md.append("Run `python3 scripts/evaluate_ml_models.py` to evaluate the models.")
        return "\n".join(md)
    
    summary = evaluation_results.get("summary", {})
    md.append("## Evaluation Summary")
    md.append("")
    md.append(f"- **Total Test Queries**: {summary.get('total_queries', 'N/A'):,}")
    md.append(f"- **Successful Evaluations**: {summary.get('successful', 'N/A'):,}")
    md.append(f"- **Failed Evaluations**: {summary.get('failed', 'N/A'):,}")
    md.append("")
    
    # Token Predictor Metrics
    md.append("## Token Predictor Performance")
    md.append("")
    token_metrics = evaluation_results.get("token_predictor", {})
    if "error" not in token_metrics and token_metrics:
        md.append(f"- **Mean Absolute Error (MAE)**: {format_number(token_metrics.get('mae', 0))} tokens")
        md.append(f"- **Root Mean Squared Error (RMSE)**: {format_number(token_metrics.get('rmse', 0))} tokens")
        md.append(f"- **Mean Absolute Percentage Error (MAPE)**: {format_number(token_metrics.get('mape', 0))}%")
        md.append(f"- **Accuracy (within 20%)**: {format_number(token_metrics.get('accuracy_within_20pct', 0))}%")
        md.append(f"- **Samples Evaluated**: {token_metrics.get('num_samples', 0):,}")
        md.append("")
        md.append("### Interpretation")
        md.append("- **MAE**: Average absolute difference between predicted and actual tokens")
        md.append("- **RMSE**: Penalizes larger errors more heavily")
        md.append("- **MAPE**: Percentage error, useful for understanding relative accuracy")
        md.append("- **Accuracy (20%)**: Percentage of predictions within 20% of actual value")
    else:
        md.append(f"*Error*: {token_metrics.get('error', 'No data available')}")
    md.append("")
    
    # Escalation Predictor Metrics
    md.append("## Escalation Predictor Performance")
    md.append("")
    escalation_metrics = evaluation_results.get("escalation_predictor", {})
    if "error" not in escalation_metrics and escalation_metrics:
        if "accuracy" in escalation_metrics:
            md.append(f"- **Accuracy**: {format_number(escalation_metrics.get('accuracy', 0))}%")
            md.append(f"- **Precision**: {format_number(escalation_metrics.get('precision', 0))}%")
            md.append(f"- **Recall**: {format_number(escalation_metrics.get('recall', 0))}%")
            md.append(f"- **F1 Score**: {format_number(escalation_metrics.get('f1_score', 0))}%")
            md.append(f"- **Samples Evaluated**: {escalation_metrics.get('num_samples', 0):,}")
            md.append("")
            
            confusion = escalation_metrics.get("confusion_matrix", {})
            if confusion:
                md.append("### Confusion Matrix")
                md.append("")
                md.append("| | Predicted: No Escalation | Predicted: Escalation |")
                md.append("| --- | --- | --- |")
                md.append(f"| **Actual: No Escalation** | {confusion.get('true_negatives', 0)} (TN) | {confusion.get('false_positives', 0)} (FP) |")
                md.append(f"| **Actual: Escalation** | {confusion.get('false_negatives', 0)} (FN) | {confusion.get('true_positives', 0)} (TP) |")
                md.append("")
        else:
            md.append(f"*Note*: {escalation_metrics.get('note', 'Requires cascading outcome analysis')}")
    else:
        md.append(f"*Error*: {escalation_metrics.get('error', 'No data available')}")
    md.append("")
    
    # Complexity Classifier Metrics
    md.append("## Complexity Classifier Performance")
    md.append("")
    complexity_metrics = evaluation_results.get("complexity_classifier", {})
    if "error" not in complexity_metrics and complexity_metrics:
        md.append(f"- **Overall Accuracy**: {format_number(complexity_metrics.get('accuracy', 0))}%")
        md.append(f"- **Samples Evaluated**: {complexity_metrics.get('num_samples', 0):,}")
        md.append("")
        
        per_class = complexity_metrics.get("per_class_accuracy", {})
        if per_class:
            md.append("### Per-Class Accuracy")
            md.append("")
            for cls, acc in sorted(per_class.items()):
                md.append(f"- **{cls.capitalize()}**: {format_number(acc)}%")
            md.append("")
        
        confusion = complexity_metrics.get("confusion_matrix", {})
        if confusion:
            md.append("### Confusion Matrix")
            md.append("")
            # Get all classes
            all_classes = set()
            for actual in confusion.keys():
                all_classes.add(actual)
                for predicted in confusion[actual].keys():
                    all_classes.add(predicted)
            all_classes = sorted(all_classes)
            
            # Header
            md.append("| Actual \\ Predicted | " + " | ".join([c.capitalize() for c in all_classes]) + " |")
            md.append("| --- | " + " | ".join(["---"] * len(all_classes)) + " |")
            
            # Rows
            for actual in all_classes:
                row = [f"**{actual.capitalize()}**"]
                for predicted in all_classes:
                    count = confusion.get(actual, {}).get(predicted, 0)
                    row.append(str(count))
                md.append("| " + " | ".join(row) + " |")
            md.append("")
    else:
        md.append(f"*Error*: {complexity_metrics.get('error', 'No data available')}")
    md.append("")
    
    # Performance Analysis
    md.append("## Performance Analysis")
    md.append("")
    md.append("### Key Insights")
    md.append("")
    
    # Analyze token predictor
    if "error" not in token_metrics and token_metrics:
        mape = token_metrics.get("mape", 100)
        accuracy_20 = token_metrics.get("accuracy_within_20pct", 0)
        
        if mape < 20:
            md.append("- ✓ **Token Predictor**: Excellent accuracy (MAPE < 20%)")
        elif mape < 30:
            md.append("- ✓ **Token Predictor**: Good accuracy (MAPE < 30%)")
        else:
            md.append("- ⚠ **Token Predictor**: Needs improvement (MAPE >= 30%)")
        
        if accuracy_20 > 80:
            md.append(f"  - {accuracy_20:.1f}% of predictions within 20% of actual")
        md.append("")
    
    # Analyze complexity classifier
    if "error" not in complexity_metrics and complexity_metrics:
        overall_acc = complexity_metrics.get("accuracy", 0)
        if overall_acc > 80:
            md.append("- ✓ **Complexity Classifier**: High accuracy (> 80%)")
        elif overall_acc > 70:
            md.append("- ✓ **Complexity Classifier**: Good accuracy (> 70%)")
        else:
            md.append("- ⚠ **Complexity Classifier**: Needs improvement (< 70%)")
        md.append("")
    
    # Recommendations
    md.append("### Recommendations")
    md.append("")
    md.append("1. **Continue Data Collection**: More training data generally improves model performance")
    md.append("2. **Monitor Production Performance**: Track model accuracy on real queries")
    md.append("3. **Retrain Periodically**: Update models as more data becomes available")
    md.append("4. **Feature Engineering**: Consider additional features if performance plateaus")
    md.append("")
    
    return "\n".join(md)


def main():
    """Generate all documentation."""
    print("=" * 80)
    print("Generating Training Documentation")
    print("=" * 80)
    print()
    
    output_dir = Path(TRAINING_CONFIG.get("output_dir", "training_data"))
    output_dir.mkdir(exist_ok=True)
    
    # Generate training summary
    print("1. Generating training summary...")
    summary_md = generate_training_summary(output_dir)
    summary_file = output_dir / "TRAINING_SUMMARY.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_md)
    print(f"   ✓ Saved: {summary_file}")
    
    # Generate performance report
    print("2. Generating performance report...")
    performance_md = generate_performance_report(output_dir)
    performance_file = output_dir / "MODEL_PERFORMANCE.md"
    with open(performance_file, 'w', encoding='utf-8') as f:
        f.write(performance_md)
    print(f"   ✓ Saved: {performance_file}")
    
    print()
    print("=" * 80)
    print("Documentation Generation Complete!")
    print("=" * 80)
    print(f"✓ Reports saved to: {output_dir}")
    print(f"   - TRAINING_SUMMARY.md")
    print(f"   - MODEL_PERFORMANCE.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
