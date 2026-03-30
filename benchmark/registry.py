from typing import Dict, Any, Type
from models.rf_model import RFModel
from models.lr_model import LRModel
from models.base_model import BaseModel

# Central registry for mapping model identifiers to their respective classes.
# This architecture allows for dynamic model instantiation and easy extension.
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "rf": RFModel,
    "lr": LRModel
}

def get_model(model_name: str, **kwargs: Any) -> BaseModel:
    """
    Factory function to retrieve and initialize a model from the registry.

    Parameters
    ----------
    model_name : str
        The unique identifier for the model (e.g., 'rf', 'lr').
    **kwargs : Any
        Hyperparameters and initialization arguments for the model.

    Returns
    -------
    BaseModel
        An initialized instance of the requested model.

    Raises
    ------
    ValueError
        If the `model_name` is not found in the registry.

    Examples
    --------
    >>> model = get_model("rf", n_estimators=100)
    >>> print(type(model))
    <class 'models.rf_model.RFModel'>
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' not found in the registry. "
            f"Currently available models: {available}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)
