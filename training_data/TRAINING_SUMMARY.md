# ML Models Training Summary

**Generated**: 2025-12-29 10:17:33

## Dataset Statistics

- **Total Original Rows**: 8,469
- **After Cleaning**: 8,398
- **Training Queries**: 6,718
- **Test Queries**: 1,680

### Cleaning Statistics
- Empty/Too Short: 0
- Too Long: 0
- Duplicates Removed: 71

### Query Length Statistics
- Minimum: 112 characters
- Maximum: 402 characters
- Average: 280.45 characters

## Data Collection Summary

- **Queries Processed**: 500
- **Successful**: 500
- **Failed**: 0

### Samples Collected
- **Token Predictions**: 630
- **Escalation Predictions**: 630
- **Complexity Predictions**: 630
- **Last Updated**: 2025-12-29T10:17:14.174817

## Model Training Status

### Token Predictor

- **Status**: ✓ Trained Successfully
- **Samples Used**: 624
- **Model Type**: XGBoost Regressor
- **Trained At**: 2025-12-29T10:16:54.544262
- **Minimum Required**: 500

### Escalation Predictor

- **Status**: ✓ Trained Successfully
- **Samples Used**: 624
- **Model Type**: XGBoost Classifier
- **Trained At**: 2025-12-29T10:16:54.574694
- **Minimum Required**: 100

### Complexity Classifier

- **Status**: ✓ Trained Successfully
- **Samples Used**: 624
- **Model Type**: XGBoost Multi-Class Classifier
- **Trained At**: 2025-12-29T10:16:54.605960
- **Minimum Required**: 100


## Model Specifications

### Token Predictor
- **Type**: XGBoost Regressor
- **Purpose**: Predict output token count for queries
- **Features**: Query length, complexity, embedding vector, context quality
- **Minimum Samples**: 500

### Escalation Predictor
- **Type**: XGBoost Classifier
- **Purpose**: Predict likelihood of needing model escalation
- **Features**: Query complexity, context quality score, query tokens, embedding
- **Minimum Samples**: 100

### Complexity Classifier
- **Type**: XGBoost Multi-Class Classifier
- **Purpose**: Classify query complexity (simple/medium/complex)
- **Features**: Query length, token count, embedding, keyword counts
- **Minimum Samples**: 100
