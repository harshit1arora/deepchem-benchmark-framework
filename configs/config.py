"""
Configuration file for the ML Benchmarking System.
Stores model lists, dataset parameters, and evaluation metrics.
"""

# List of models to run in the benchmark
MODELS_TO_RUN = ["rf", "lr"]

# Dataset configuration
DATASET_CONFIG = {
    "n_samples": 1000,
    "n_features": 20,
    "random_state": 42,
    "test_size": 0.2
}

# Metrics to evaluate
METRICS = ["accuracy", "roc_auc"]
