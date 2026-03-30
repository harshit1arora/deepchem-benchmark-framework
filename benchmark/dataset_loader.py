from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from .logger import logger
from .dataset_adapter import DatasetAdapter

class DatasetLoader:
    """
    Enhanced DatasetLoader with DeepChem (MolNet) support.
    
    Handles loading of synthetic data and molecular datasets (e.g., Tox21) 
    using a unified interface.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the loader with dataset configuration parameters.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration containing 'type', 'n_samples', 'test_size', etc.
        """
        self.config = config
        self.adapter = DatasetAdapter()
        self.tasks: List[str] = []
        logger.debug(f"DatasetLoader initialized with config: {self.config}")

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Unified method to load and prepare training/testing datasets.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple of (X_train, X_test, y_train, y_test).
        """
        dataset_type = self.config.get("type", "synthetic").lower()
        
        if dataset_type == "synthetic":
            return self._load_synthetic()
        elif dataset_type == "tox21":
            return self._load_deepchem_molnet("tox21")
        else:
            # Flexible for future MolNet datasets
            msg = f"Dataset type '{dataset_type}' is not yet supported or is invalid."
            logger.error(msg)
            raise ValueError(msg)

    def _load_synthetic(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load scikit-learn synthetic data."""
        logger.info("Generating synthetic classification dataset...")
        X, y = make_classification(
            n_samples=self.config.get("n_samples", 1000),
            n_features=self.config.get("n_features", 20),
            random_state=self.config.get("random_state", 42)
        )
        self.tasks = ["synthetic_task"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.get("test_size", 0.2),
            random_state=self.config.get("random_state", 42)
        )
        return X_train, X_test, y_train, y_test

    def _load_deepchem_molnet(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load datasets from DeepChem's MolNet collection.
        
        Currently supports 'tox21' and is extensible for others.
        """
        try:
            import deepchem as dc
            logger.info(f"Loading DeepChem MolNet dataset: {dataset_name}...")
            
            # Load the dataset using MolNet
            # featurizer='ECFP' (Extended Connectivity Fingerprints) is a common default for molecular models
            if dataset_name == "tox21":
                tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='ECFP')
            else:
                raise ValueError(f"MolNet loader for '{dataset_name}' not specifically implemented.")

            self.tasks = tasks
            train_ds, valid_ds, test_ds = datasets
            
            logger.info(f"Dataset loaded. Tasks: {len(self.tasks)}. Train size: {len(train_ds)}.")

            # Convert to numpy via adapter
            X_train, y_train, _ = self.adapter.to_numpy(train_ds)
            X_test, y_test, _ = self.adapter.to_numpy(test_ds)
            
            # Handle potential multi-task NaNs (common in molecular data)
            X_train, y_train = self.adapter.handle_missing_values(X_train, y_train)
            X_test, y_test = self.adapter.handle_missing_values(X_test, y_test)

            return X_train, X_test, y_train, y_test

        except ImportError:
            msg = "DeepChem is not installed. Please install it using 'pip install deepchem'."
            logger.error(msg)
            raise ImportError(msg)
        except Exception as e:
            msg = f"Failed to load DeepChem dataset '{dataset_name}': {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e
            
    def get_tasks(self) -> List[str]:
        """Return the list of tasks for the loaded dataset."""
        return self.tasks
