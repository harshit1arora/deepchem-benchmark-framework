import argparse
import json
import os
import sys
from typing import Dict, Any
from benchmark.runner import BenchmarkRunner
from benchmark.utils import pretty_print_results, validate_config
from benchmark.logger import logger

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load benchmarking configuration from a JSON file.
    
    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to parse configuration file: {str(e)}")
        sys.exit(1)

def main():
    """
    Entry point for the production-grade ML Benchmarking Framework.
    
    Handles CLI arguments, loads configuration, and executes the benchmark.
    """
    parser = argparse.ArgumentParser(description="DeepChem ML Benchmarking Framework")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.json",
        help="Path to the JSON configuration file (default: configs/config.json)"
    )
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Override models to run (e.g., --models rf lr)"
    )
    
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Initializing DeepChem ML Benchmarking Framework")
    logger.info("=" * 50)

    # Load configuration
    config = load_config(args.config)
    
    # Override models if provided via CLI
    if args.models:
        config["models"] = args.models
    
    try:
        # Validate the configuration before execution
        validate_config(config)
        
        # Extract validated parameters
        models_to_run = config["models"]
        dataset_config = config["dataset"]
        metrics = config["metrics"]

        # Initialize and execute the benchmark
        runner = BenchmarkRunner(
            models=models_to_run,
            dataset_config=dataset_config,
            metrics=metrics
        )
        results = runner.run()

        # Display results in a structured table
        print("\n" + "=" * 20 + " BENCHMARK RESULTS " + "=" * 20)
        pretty_print_results(results)
        
        logger.info("Benchmarking process completed successfully.")
        
    except ValueError as e:
        logger.error(f"Configuration Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Benchmark execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
