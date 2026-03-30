# DeepChem Benchmarking Suite 

[![GSoC 2026](https://img.shields.io/badge/GSoC-2026-blue.svg)](https://summerofcode.withgoogle.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
The **DeepChem Benchmarking Suite** is a modular, production-grade framework designed to evaluate and compare machine learning models on molecular datasets. Built with scalability and extensibility in mind, this project serves as a foundation for a **Google Summer of Code (GSoC)** proposal to integrate standardized benchmarking into the [DeepChem](https://github.com/deepchem/deepchem) ecosystem.

The system allows researchers to run multiple models (e.g., Random Forest, Logistic Regression) against various datasets (scikit-learn synthetic, Tox21, etc.) using a unified evaluation pipeline, ensuring consistent metrics and reproducible results.

---

## Key Features
- **Clean & Modular Architecture**: Separation of concerns between model definitions, data loading, evaluation, and orchestration.
- **DeepChem Native Integration**: Full support for `deepchem.molnet` datasets and specialized molecular featurizers.
- **Multi-Task Support**: Automatically handles multi-task labels using macro-averaging and multi-output classifiers.
- **Production-Quality Logging**: Comprehensive logging system with both console and file handlers.
- **Config-Driven Execution**: Entire benchmarks can be defined via JSON configurations with built-in validation.
- **CLI Support**: Execute experiments and override parameters directly from the terminal.
- **Experiment Tracking**: Automatic generation of unique Experiment IDs with results saved in both JSON and CSV formats.
- **Type Safety**: Fully type-hinted codebase with **numpydoc** documentation.

---

## Project Structure
```text
deepchem-benchmark-suite/
│
├── benchmark/              # Core benchmarking logic
│   ├── runner.py           # Main orchestrator (Pipeline: Load -> Train -> Eval)
│   ├── registry.py         # Dynamic model factory
│   ├── evaluator.py        # Metric computation (Acc, ROC-AUC, Precision, Recall)
│   ├── dataset_loader.py   # Data generation and splitting
│   ├── dataset_adapter.py  # Data normalization and NaN cleaning
│   ├── logger.py           # Structured logging system
│   └── utils.py            # Result persistence and config validation
│
├── models/                 # Model implementations
│   ├── base_model.py       # Abstract Base Class defining the interface
│   ├── rf_model.py         # Random Forest implementation
│   └── lr_model.py         # Logistic Regression implementation
│
├── configs/                # Experiment configurations
│   ├── config.json         # Standard experiment definition
│   └── tox21_config.json   # DeepChem molecular experiment definition
│
├── results/                # Persistent experiment outputs (JSON/CSV)
├── logs/                   # Detailed execution logs
├── tests/                  # Automated test suite
├── main.py                 # CLI entry point
└── requirements.txt        # Project dependencies
```

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/deepchem-benchmark-suite.git
   cd deepchem-benchmark-suite
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage

### Running a Benchmark
Execute the default benchmark using the provided configuration:
```bash
python main.py
```

### Running a DeepChem Molecular Benchmark
Run a specialized experiment on the **Tox21** dataset:
```bash
python main.py --config configs/tox21_config.json
```

### Expected Output Example
```text
2026-03-30 17:30:14 - deepchem_benchmark - INFO - Initializing DeepChem ML Benchmarking Framework
2026-03-30 17:30:14 - deepchem_benchmark - INFO - Loaded configuration from configs/config.json
2026-03-30 17:30:14 - deepchem_benchmark - INFO - Starting benchmark execution...
...
==================== BENCHMARK RESULTS ====================
-------------------------------------------------------------------
Model           | Accuracy   | Roc_auc    | Precision  | Recall
-------------------------------------------------------------------
rf              | 0.8900     | 0.9399     | 0.9570     | 0.8318
lr              | 0.8550     | 0.9216     | 0.9149     | 0.8037
-------------------------------------------------------------------
```

---

## Testing
The suite includes a comprehensive set of tests covering the runner, evaluator, and registry.
```bash
# Set PYTHONPATH and run tests
$env:PYTHONPATH="." 
pytest tests/test_runner.py
```

---

## 🤝 Contributing
Contributions are welcome! If you're interested in improving the benchmarking framework or adding new models/datasets, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License
This project is licensed under the MIT License.

---
**Developed for the DeepChem GSoC 2026 Proposal by Harshit Arora.**
