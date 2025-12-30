# Customer Support Dataset Training Pipeline

Complete pipeline to prepare, collect data, train, and evaluate ML models using the customer support ticket dataset.

## Overview

This pipeline:
1. **Prepares** the dataset (cleans, deduplicates, splits 80:20)
2. **Collects** training data by running queries through the platform
3. **Trains** all 3 ML models (Token Predictor, Escalation Predictor, Complexity Classifier)
4. **Evaluates** trained models on test set

## Quick Start

### Run Complete Pipeline

```bash
python3 scripts/run_training_pipeline.py
```

### Run Individual Steps

1. **Prepare Dataset**:
   ```bash
   python3 scripts/prepare_customer_support_dataset.py
   ```

2. **Collect Training Data** (run 1000 queries):
   ```bash
   python3 scripts/collect_training_data.py --sample-size 1000
   ```

3. **Train Models**:
   ```bash
   python3 scripts/train_ml_models.py
   ```

4. **Evaluate Models**:
   ```bash
   python3 scripts/evaluate_ml_models.py --max-queries 100
   ```

## Configuration

Edit `scripts/training_config.py` to adjust:
- Sample size for data collection
- Min samples required for training
- Train/test split ratio
- Query length filters

## Output Files

All outputs are saved to `training_data/` directory:

- `training_queries.json` - 80% training queries (cleaned)
- `test_queries.json` - 20% test queries (cleaned)
- `dataset_stats.json` - Dataset statistics
- `checkpoint.json` - Data collection checkpoint (for resuming)
- `training_results.json` - Model training results
- `evaluation_report.json` - Evaluation metrics (JSON)
- `evaluation_report.md` - Evaluation report (human-readable)

## Resuming Data Collection

If data collection is interrupted, it automatically resumes from checkpoint:

```bash
python3 scripts/collect_training_data.py
```

To start fresh:

```bash
python3 scripts/collect_training_data.py --no-resume
```

## Data Requirements

- **Token Predictor**: Needs ~500 samples
- **Escalation Predictor**: Needs ~100 samples
- **Complexity Classifier**: Needs ~100 samples

The data collection script will show readiness status after each run.

## Notes

- Data collection requires API access (OpenAI/Gemini)
- Running 1000 queries may take time and incur API costs
- Checkpoint system allows safe interruption/resume
- All data is stored in unified `ml_training_data.db` database



