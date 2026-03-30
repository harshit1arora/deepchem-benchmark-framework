from sklearn.ensemble import RandomForestClassifier
from typing import Any, Dict
from .base_model import BaseModel
import numpy as np

class RFModel(BaseModel):
    """
    Random Forest Model wrapper with native multi-task support.
    
    scikit-learn's RandomForestClassifier handles multi-task 
    datasets natively when y is a 2D array.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize Random Forest model.
        """
        self.model = RandomForestClassifier(**kwargs)
        self.name = "RandomForest"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model, handling both single and multi-task labels.
        """
        # Ensure y is 2D for multi-task support in scikit-learn
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels.
        """
        preds = self.model.predict(X)
        # Handle both single-task (ravel to 1D) and multi-task (stay 2D)
        if preds.ndim == 2 and preds.shape[1] == 1:
            return preds.ravel()
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for multi-task binary classification.
        """
        probas_list = self.model.predict_proba(X)
        
        # If it's multi-output, predict_proba returns a list of (n_samples, n_classes) arrays
        if isinstance(probas_list, list):
            # For each task, extract the positive class probability (last column)
            probas = np.column_stack([p[:, -1] for p in probas_list])
            return probas.ravel() if probas.shape[1] == 1 else probas
        else:
            # Single task case (n_samples, n_classes)
            return probas_list[:, -1]

    def get_name(self) -> str:
        return self.name

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()
