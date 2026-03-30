from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from typing import Any, Dict
from .base_model import BaseModel
import numpy as np

class LRModel(BaseModel):
    """
    Logistic Regression Model wrapper with multi-task support.
    
    Uses scikit-learn's MultiOutputClassifier to handle multi-task 
    datasets like Tox21.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize Logistic Regression model.
        """
        base_model = LogisticRegression(max_iter=1000, **kwargs)
        self.model = MultiOutputClassifier(base_model)
        self.name = "LogisticRegression"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model, handling both single and multi-task labels.
        """
        # MultiOutputClassifier requires y to be at least 2D
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
        Predict probabilities for multi-task binary classification.
        """
        probas_list = self.model.predict_proba(X)
        
        # MultiOutputClassifier always returns a list of (n_samples, n_classes) arrays
        if isinstance(probas_list, list):
            # For each task, extract the positive class probability (last column)
            probas = np.column_stack([p[:, -1] for p in probas_list])
            return probas.ravel() if probas.shape[1] == 1 else probas
        else:
            return probas_list[:, -1]

    def get_name(self) -> str:
        return self.name

    def get_params(self) -> Dict[str, Any]:
        return self.model.get_params()
