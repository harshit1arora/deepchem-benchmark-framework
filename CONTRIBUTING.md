# Contributing to DeepChem Benchmarking Suite 

Welcome! We are excited that you're interested in contributing to this project. This suite is designed to be highly modular and extensible, making it easy to add new models, datasets, and metrics.

## How to Add a New Model

To integrate a new model into the framework, follow these steps:

1.  **Create the Model Wrapper**:
    *   Add a new file in the `models/` directory (e.g., `models/svm_model.py`).
    *   Inherit from the `BaseModel` class.
    *   Implement the `fit()`, `predict()`, `predict_proba()`, `get_name()`, and `get_params()` methods.

    ```python
    from models.base_model import BaseModel
    from sklearn.svm import SVC
    import numpy as np

    class SVMModel(BaseModel):
        def __init__(self, **kwargs):
            self.model = SVC(probability=True, **kwargs)
            self.name = "SVM"

        def fit(self, X, y):
            self.model.fit(X, y)

        def predict(self, X):
            return self.model.predict(X)

        def predict_proba(self, X):
            return self.model.predict_proba(X)[:, 1]

        def get_name(self):
            return self.name

        def get_params(self):
            return self.model.get_params()
    ```

2.  **Register the Model**:
    *   Open `benchmark/registry.py`.
    *   Import your new model class and add it to the `MODEL_REGISTRY` dictionary.

    ```python
    from models.svm_model import SVMModel

    MODEL_REGISTRY = {
        "rf": RFModel,
        "lr": LRModel,
        "svm": SVMModel  # Add your new model here
    }
    ```

## How to Add a New Dataset

To support a new dataset from DeepChem or elsewhere:

1.  **Update the Loader**:
    *   Open `benchmark/dataset_loader.py`.
    *   Add a new private method (e.g., `_load_esol()`) to fetch and split the data.
    *   Update the `load_data()` method to handle the new dataset identifier.

2.  **Ensure Compatibility**:
    *   Use the `DatasetAdapter` to convert the data into consistent numpy arrays.
    *   Verify that multi-task labels are handled correctly if applicable.

## Coding Standards

*   **Documentation**: Use **numpydoc** style for all classes and functions.
*   **Type Hinting**: Always include type hints for function parameters and return values.
*   **Logging**: Use the project's centralized logger (`benchmark.logger`) instead of `print()`.
*   **Testing**: Add unit tests for any new features in the `tests/` directory.

## Running Tests

Before submitting a pull request, ensure all tests pass:

```bash
$env:PYTHONPATH="."
pytest tests/test_runner.py
```

Thank you for helping us build a robust benchmarking framework for molecular machine learning!
