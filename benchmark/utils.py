import json
import csv
import os
from typing import Dict, Any, List
from .logger import logger
from .registry import MODEL_REGISTRY

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the benchmarking configuration dictionary.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing 'models', 'dataset', and 'metrics'.

    Raises
    ------
    ValueError
        If required fields are missing or contain invalid values.
    """
    # 1. Check for required top-level fields
    required_fields = ["models", "dataset", "metrics"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: '{field}'")

    # 2. Validate Models
    models = config["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("'models' must be a non-empty list of model identifiers.")
    
    for model_id in models:
        if model_id not in MODEL_REGISTRY:
            raise ValueError(
                f"Invalid model identifier: '{model_id}'. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )

    # 3. Validate Dataset
    dataset = config["dataset"]
    if not isinstance(dataset, (dict, str)):
        raise ValueError("'dataset' must be a dictionary or a string identifier.")
    
    # If it's a dict, check for 'type'
    if isinstance(dataset, dict) and "type" not in dataset:
        raise ValueError("Dataset configuration dictionary must include a 'type' field.")

    # 4. Validate Metrics
    metrics = config["metrics"]
    supported_metrics = ["accuracy", "roc_auc", "precision", "recall"]
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("'metrics' must be a non-empty list.")
    
    for metric in metrics:
        if metric not in supported_metrics:
            raise ValueError(
                f"Unsupported metric: '{metric}'. "
                f"Supported metrics: {supported_metrics}"
            )

    logger.info("Configuration validated successfully.")

def pretty_print_results(results: Dict[str, Dict[str, float]]) -> None:
    """
    Format and display benchmarking results in a clean table format.

    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Dictionary mapping model names to their computed metrics.
    """
    if not results:
        logger.warning("No results to display.")
        return

    # Dynamically determine headers from the first model's metrics
    metrics_list = list(next(iter(results.values())).keys())
    
    header = f"{'Model':<15}"
    for m in metrics_list:
        header += f" | {m.capitalize():<10}"
    
    divider = "-" * len(header)
    
    print(divider)
    print(header)
    print(divider)

    for model, metrics in results.items():
        row = f"{model:<15}"
        for m in metrics_list:
            val = metrics.get(m, 0.0)
            row += f" | {val:<10.4f}"
        print(row)
    
    print(divider)

def save_results(results: Dict[str, Dict[str, float]], 
                 experiment_id: str, 
                 output_dir: str = "results") -> None:
    """
    Save benchmarking results to JSON and CSV formats.

    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Dictionary of computed metrics per model.
    experiment_id : str
        Unique identifier for the current run.
    output_dir : str, optional
        Directory where results will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.debug(f"Created output directory: {output_dir}")

    # Save to JSON
    json_path = os.path.join(output_dir, f"results_{experiment_id}.json")
    try:
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to JSON: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON results: {str(e)}")

    # Save to CSV
    csv_path = os.path.join(output_dir, f"results_{experiment_id}.csv")
    try:
        if results:
            metrics_list = list(next(iter(results.values())).keys())
            headers = ["model"] + metrics_list
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for model, metrics in results.items():
                    row = [model] + [metrics.get(m, 0.0) for m in metrics_list]
                    writer.writerow(row)
            logger.info(f"Results saved to CSV: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV results: {str(e)}")
