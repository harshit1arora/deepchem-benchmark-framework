from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from .dataset_loader import DatasetLoader
from .registry import get_model
from .evaluator import Evaluator
from .logger import logger
from .utils import save_results
from configs.config import MODELS_TO_RUN, DATASET_CONFIG, METRICS

class BenchmarkRunner:
    """
    Orchestrator for the ML benchmarking framework.

    This class manages the lifecycle of a benchmarking experiment, including 
    data loading, model initialization, training, evaluation, and result 
    persistence. It supports both standard scikit-learn workflows and 
    specialized DeepChem molecular datasets.

    Parameters
    ----------
    models : List[str], optional
        Names of models to include in the benchmark. Defaults to 
        `configs.config.MODELS_TO_RUN`.
    dataset_config : Dict[str, Any], optional
        Configuration parameters for the dataset loader. Defaults to 
        `configs.config.DATASET_CONFIG`.
    metrics : List[str], optional
        List of metrics to evaluate. Defaults to `configs.config.METRICS`.

    Examples
    --------
    >>> runner = BenchmarkRunner(models=["rf", "lr"])
    >>> results = runner.run()
    """

    def __init__(self, 
                 models: Optional[List[str]] = None, 
                 dataset_config: Optional[Dict[str, Any]] = None,
                 metrics: Optional[List[str]] = None) -> None:
        """Initialize the BenchmarkRunner with experiment settings."""
        self.models_to_run = models or MODELS_TO_RUN
        self.dataset_config = dataset_config or DATASET_CONFIG
        self.metrics_to_eval = metrics or METRICS
        
        self.loader = DatasetLoader(self.dataset_config)
        self.evaluator = Evaluator(self.metrics_to_eval)
        self.results: Dict[str, Dict[str, float]] = {}
        
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Initialized BenchmarkRunner (Experiment ID: {self.experiment_id})")

    def run(self) -> Dict[str, Dict[str, float]]:
        """
        Execute the benchmarking pipeline for all specified models.

        Returns
        -------
        Dict[str, Dict[str, float]]
            A dictionary mapping each model name to its performance metrics.

        Raises
        ------
        RuntimeError
            If data loading or model execution fails critically.
        """
        logger.info("Starting benchmark execution...")
        
        try:
            # Step 1: Data Acquisition
            X_train, X_test, y_train, y_test = self.loader.load_data()
            tasks = self.loader.get_tasks()
            
            logger.info(f"Dataset Loaded: {len(X_train)} train, {len(X_test)} test samples.")
            logger.info(f"Tasks detected: {len(tasks)}")
                
        except Exception as e:
            msg = f"Data loading failed: {str(e)}"
            logger.critical(msg)
            raise RuntimeError(msg) from e

        if X_train.size == 0 or X_test.size == 0:
            logger.warning("Empty dataset. Skipping benchmark.")
            return {}

        # Step 2: Model Benchmarking Loop
        for model_key in self.models_to_run:
            logger.info(f"--- Benchmarking Model: {model_key} ---")
            
            try:
                # 2.1: Model Instantiation
                model = get_model(model_key)
                model_display_name = model.get_name()
                
                # 2.2: Training
                logger.info(f"Training {model_display_name}...")
                model.fit(X_train, y_train)
                
                # 2.3: Inference
                logger.info(f"Predicting with {model_display_name}...")
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                
                # 2.4: Metric Evaluation
                logger.info(f"Evaluating {model_display_name}...")
                model_results = self.evaluator.evaluate(y_test, y_pred, y_proba)
                self.results[model_key] = model_results
                
                logger.info(f"Results for {model_display_name}: {model_results}")
                
            except Exception as e:
                logger.error(f"Benchmark failed for model '{model_key}': {str(e)}")
                continue

        # Step 3: Result Finalization
        if self.results:
            save_results(self.results, self.experiment_id)
            logger.info("Benchmark execution successfully finalized.")
        else:
            logger.warning("No valid results were generated.")
            
        return self.results
