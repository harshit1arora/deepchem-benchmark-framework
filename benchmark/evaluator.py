from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from typing import Dict, List, Any
import numpy as np
from .logger import logger

class Evaluator:
    """
    Performance evaluator for machine learning models.

    This class computes and averages classification metrics across multiple 
    tasks. It is designed to be compatible with both single-task and 
    multi-task datasets, providing macro-averaged results.

    Examples
    --------
    >>> evaluator = Evaluator(metrics=["accuracy", "roc_auc"])
    >>> results = evaluator.evaluate(y_true, y_pred, y_proba)
    >>> print(results)
    {'accuracy': 0.85, 'roc_auc': 0.91}
    """

    def __init__(self, metrics: List[str]) -> None:
        """
        Initialize the evaluator with a specific set of metrics.

        Parameters
        ----------
        metrics : List[str]
            Names of metrics to calculate (e.g., 'accuracy', 'roc_auc', 
            'precision', 'recall').
        """
        self.metrics = metrics
        self.supported_metrics = {
            "accuracy": accuracy_score,
            "roc_auc": roc_auc_score,
            "precision": precision_score,
            "recall": recall_score
        }
        self._validate_metrics()
        logger.debug(f"Evaluator initialized with metrics: {self.metrics}")

    def _validate_metrics(self) -> None:
        """Ensure all requested metrics are supported."""
        for m in self.metrics:
            if m not in self.supported_metrics:
                msg = f"Unsupported metric: '{m}'. Supported: {list(self.supported_metrics.keys())}"
                logger.error(msg)
                raise ValueError(msg)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance and return averaged metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True target labels of shape (n_samples, n_tasks).
        y_pred : np.ndarray
            Predicted labels of shape (n_samples, n_tasks).
        y_proba : np.ndarray
            Predicted class probabilities of shape (n_samples, n_tasks).

        Returns
        -------
        Dict[str, float]
            A dictionary containing macro-averaged performance metrics.
        """
        if y_true.size == 0:
            logger.warning("Empty ground truth provided for evaluation.")
            return {m: 0.0 for m in self.metrics}

        # Normalize to 2D for multi-task consistency
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_proba.ndim == 1:
            y_proba = y_proba.reshape(-1, 1)

        n_tasks = y_true.shape[1]
        task_metrics = {metric: [] for metric in self.metrics}
        
        for i in range(n_tasks):
            y_t = y_true[:, i]
            y_p = y_pred[:, i]
            y_pb = y_proba[:, i]
            
            # Filter NaNs for the current task
            mask = ~np.isnan(y_t)
            if not np.any(mask):
                continue
                
            y_t_f, y_p_f, y_pb_f = y_t[mask], y_p[mask], y_pb[mask]
            
            for metric in self.metrics:
                try:
                    val = self._calculate_metric(metric, y_t_f, y_p_f, y_pb_f)
                    task_metrics[metric].append(val)
                except Exception as e:
                    # Specific tasks might fail (e.g., only one class present for ROC-AUC)
                    logger.debug(f"Task {i} evaluation failed for {metric}: {str(e)}")
                    continue
        
        # Macro-average across tasks
        avg_results = {}
        for metric, values in task_metrics.items():
            avg_results[metric] = float(np.mean(values)) if values else 0.0
            
        return avg_results

    def _calculate_metric(self, metric: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
        """Internal metric calculation logic."""
        if metric == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        elif metric == "roc_auc":
            # ROC-AUC requires both classes to be present
            if len(np.unique(y_true)) < 2:
                raise ValueError("ROC-AUC requires both positive and negative samples.")
            return float(roc_auc_score(y_true, y_proba))
        elif metric == "precision":
            return float(precision_score(y_true, y_pred, zero_division=0))
        elif metric == "recall":
            return float(recall_score(y_true, y_pred, zero_division=0))
        return 0.0
