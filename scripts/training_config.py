"""Configuration for customer support dataset training pipeline."""

TRAINING_CONFIG = {
    "dataset_path": "ML Model Training dataset/customer_support_tickets.csv",
    "query_column": "Ticket Description",
    "product_column": "Product Purchased",
    "train_split": 0.8,
    "test_split": 0.2,
    "random_seed": 42,
    "sample_size": 1000,  # Number of queries to run for data collection
    "min_query_length": 10,
    "max_query_length": 5000,
    "checkpoint_interval": 100,
    "batch_size": 10,
    "min_samples": {
        "token_predictor": 500,
        "escalation_predictor": 100,
        "complexity_classifier": 100,
    },
    "output_dir": "training_data",
}



