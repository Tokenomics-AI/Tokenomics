# ML Models Performance Report

**Generated**: 2025-12-29 10:17:33

## Evaluation Summary

- **Total Test Queries**: 1,680
- **Successful Evaluations**: 1
- **Failed Evaluations**: 1,679

## Token Predictor Performance

- **Mean Absolute Error (MAE)**: 249.00 tokens
- **Root Mean Squared Error (RMSE)**: 249.00 tokens
- **Mean Absolute Percentage Error (MAPE)**: 64.18%
- **Accuracy (within 20%)**: 0.00%
- **Samples Evaluated**: 1

### Interpretation
- **MAE**: Average absolute difference between predicted and actual tokens
- **RMSE**: Penalizes larger errors more heavily
- **MAPE**: Percentage error, useful for understanding relative accuracy
- **Accuracy (20%)**: Percentage of predictions within 20% of actual value

## Escalation Predictor Performance

*Note*: Requires cascading outcome analysis

## Complexity Classifier Performance

- **Overall Accuracy**: 100.00%
- **Samples Evaluated**: 1

### Per-Class Accuracy

- **Medium**: 100.00%

### Confusion Matrix

| Actual \ Predicted | Medium |
| --- | --- |
| **Medium** | 1 |


## Performance Analysis

### Key Insights

- ⚠ **Token Predictor**: Needs improvement (MAPE >= 30%)

- ✓ **Complexity Classifier**: High accuracy (> 80%)

### Recommendations

1. **Continue Data Collection**: More training data generally improves model performance
2. **Monitor Production Performance**: Track model accuracy on real queries
3. **Retrain Periodically**: Update models as more data becomes available
4. **Feature Engineering**: Consider additional features if performance plateaus
