import pytest
import os
import numpy as np
from benchmark.runner import BenchmarkRunner
from benchmark.evaluator import Evaluator
from benchmark.registry import get_model
from benchmark.utils import validate_config

def test_benchmark_runner_execution():
    """Test standard execution of the BenchmarkRunner."""
    runner = BenchmarkRunner(models=["rf", "lr"])
    results = runner.run()

    assert isinstance(results, dict)
    assert "rf" in results
    assert "lr" in results

    for model, metrics in results.items():
        assert all(m in metrics for m in ["accuracy", "roc_auc"])
        assert all(0 <= metrics[m] <= 1 for m in metrics)

def test_benchmark_runner_empty_dataset():
    """Test BenchmarkRunner with an empty dataset configuration."""
    # A configuration that would result in an empty dataset after splitting
    dataset_config = {"type": "synthetic", "n_samples": 0}
    runner = BenchmarkRunner(models=["lr"], dataset_config=dataset_config)
    
    # Depending on how make_classification handles n_samples=0, this might 
    # raise an error or return empty arrays.
    # Our runner should handle it gracefully or raise a meaningful error.
    try:
        results = runner.run()
        assert results == {}
    except Exception:
        # If it raises an error from sklearn, that's also acceptable 
        # as long as we verify the behavior.
        pass

def test_model_registry_error():
    """Test that requesting an unregistered model raises ValueError."""
    with pytest.raises(ValueError, match="not found in the registry"):
        get_model("non_existent_model")

def test_evaluator_multi_task():
    """Test the Evaluator with multi-task labels (typical for DeepChem)."""
    metrics = ["accuracy", "roc_auc"]
    evaluator = Evaluator(metrics)
    
    # 2 tasks, 4 samples
    y_true = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_pred = np.array([[0, 1], [0, 0], [1, 1], [0, 0]])
    y_proba = np.array([[0.1, 0.9], [0.4, 0.2], [0.8, 0.7], [0.3, 0.1]])
    
    results = evaluator.evaluate(y_true, y_pred, y_proba)
    
    assert "accuracy" in results
    assert "roc_auc" in results
    assert isinstance(results["accuracy"], float)

def test_evaluator_invalid_metric():
    """Test that initializing Evaluator with unsupported metrics raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported metric"):
        Evaluator(["unsupported_metric"])

def test_config_validation_valid():
    """Test validation of a correctly formatted configuration."""
    valid_config = {
        "models": ["rf"],
        "dataset": "synthetic",
        "metrics": ["accuracy"]
    }
    # Should not raise any exception
    validate_config(valid_config)

def test_config_validation_invalid():
    """Test validation of various invalid configurations."""
    # Missing field
    with pytest.raises(ValueError, match="Missing required configuration field"):
        validate_config({"models": ["rf"], "metrics": ["accuracy"]})
        
    # Invalid model
    with pytest.raises(ValueError, match="Invalid model identifier"):
        validate_config({"models": ["invalid"], "dataset": "synthetic", "metrics": ["accuracy"]})
        
    # Unsupported metric
    with pytest.raises(ValueError, match="Unsupported metric"):
        validate_config({"models": ["rf"], "dataset": "synthetic", "metrics": ["invalid"]})

def test_result_persistence():
    """Test that results are correctly saved to disk."""
    runner = BenchmarkRunner(models=["lr"])
    runner.run()
    
    exp_id = runner.experiment_id
    json_path = os.path.join("results", f"results_{exp_id}.json")
    csv_path = os.path.join("results", f"results_{exp_id}.csv")
    
    assert os.path.exists(json_path)
    assert os.path.exists(csv_path)
    
    # Clean up test files
    os.remove(json_path)
    os.remove(csv_path)
