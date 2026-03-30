from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    This interface ensures that every model implements the necessary 
    methods for training, prediction, and inspection, following 
    the scikit-learn convention while being extensible for DeepChem.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.
        
        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, n_features).
        y : np.ndarray
            Training labels of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Features to predict on of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for classification tasks.
        
        Parameters
        ----------
        X : np.ndarray
            Features to predict on of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Predicted probabilities for the positive class.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the model.
        
        Returns
        -------
        str
            The model's name.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Return the model's hyperparameters.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of model parameters.
        """
        pass
