import numpy as np
from typing import Tuple, List, Optional, Any
from .logger import logger

class DatasetAdapter:
    """
    Adapter layer to normalize different dataset formats (scikit-learn, DeepChem).

    This class ensures that data from various sources is consistent and ready 
    for model training and evaluation, handling multi-task labels and 
    cleaning missing values common in molecular datasets.

    Examples
    --------
    >>> adapter = DatasetAdapter()
    >>> X, y, tasks = adapter.to_numpy(deepchem_dataset)
    >>> X_clean, y_clean = adapter.handle_missing_values(X, y)
    """

    @staticmethod
    def to_numpy(dataset: Any) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert a dataset object to consistent numpy arrays.

        Parameters
        ----------
        dataset : Any
            The dataset object to convert. Supports DeepChem Dataset objects 
            or (X, y) tuples.

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Label matrix or vector of shape (n_samples, n_tasks).
        tasks : List[str]
            List of task names associated with the labels.

        Raises
        ------
        ValueError
            If the dataset format is not recognized.
        """
        # Handle DeepChem Dataset objects (DiskDataset, NumpyDataset, etc.)
        if hasattr(dataset, 'X') and hasattr(dataset, 'y'):
            X = dataset.X
            y = dataset.y
            # Retrieve task names if available (common in MolNet datasets)
            tasks = getattr(dataset, 'tasks', ["task_0"])
            
            # Normalize y shape: ensure it's at least 2D for consistency in multi-task logic
            if y.ndim == 1:
                y = y.reshape(-1, 1)
                
            return X, y, tasks
        
        # Handle standard (X, y) tuples
        elif isinstance(dataset, tuple) and len(dataset) >= 2:
            X, y = np.array(dataset[0]), np.array(dataset[1])
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            return X, y, ["task_0"]
            
        else:
            msg = f"Unsupported dataset format: {type(dataset)}. Expected DeepChem Dataset or (X, y) tuple."
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def handle_missing_values(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean missing values (NaNs) from features and labels.

        Molecular datasets often contain NaNs in labels where a specific 
        assay was not performed for a molecule.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Label matrix.

        Returns
        -------
        X_cleaned : np.ndarray
            Cleaned feature matrix.
        y_cleaned : np.ndarray
            Cleaned label matrix.
        """
        if X.size == 0 or y.size == 0:
            logger.warning("Empty dataset provided to handle_missing_values.")
            return X, y

        # Check for NaNs in labels (very common in Tox21, etc.)
        nan_mask_y = np.isnan(y)
        if np.any(nan_mask_y):
            # For multi-task, we keep the sample if at least one task is labeled
            # However, for simple scikit-learn models, we often need to filter 
            # or impute. Here we filter samples that have NO valid labels.
            valid_samples_mask = ~np.all(nan_mask_y, axis=1)
            X = X[valid_samples_mask]
            y = y[valid_samples_mask]
            logger.info(f"Filtered samples with no valid labels. Remaining: {len(X)}")

        # Check for NaNs in features
        nan_mask_X = np.isnan(X)
        if np.any(nan_mask_X):
            logger.warning("NaN values detected in features. Samples will be filtered.")
            valid_features_mask = ~np.any(nan_mask_X, axis=1)
            X = X[valid_features_mask]
            y = y[valid_features_mask]
            logger.info(f"Filtered samples with invalid features. Remaining: {len(X)}")
            
        return X, y
