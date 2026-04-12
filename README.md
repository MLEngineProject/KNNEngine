# KNN Engine Core

[![PyPI version](https://img.shields.io/pypi/v/knn-engine-core?logo=pypi&logoColor=white)](https://pypi.org/project/knn-engine-core/)
[![Python](https://img.shields.io/pypi/pyversions/knn-engine-core?logo=python&logoColor=white)](https://pypi.org/project/knn-engine-core/)
[![Build system](https://img.shields.io/badge/build-scikit--build--core-blue?logo=cmake&logoColor=white)](https://scikit-build-core.readthedocs.io/)
[![Bindings](https://img.shields.io/badge/bindings-pybind11-4C72B0?logo=python&logoColor=white)](https://pybind11.readthedocs.io/)

High-performance K-Nearest Neighbors (KNN) classification engine implemented in C++17, with adaptive Principal Component Analysis (PCA) for dimensionality reduction, and exposed to Python via pybind11.

This project is designed for fast experimentation from Python while keeping the compute-intensive path in native C++.

## Highlights

- C++17 implementation of KNN with efficient top-k neighbor selection (`std::nth_element`).
- Adaptive PCA that supports:
  - variance-threshold mode (for example `0.95` keeps enough components to preserve 95% variance),
  - fixed-component mode (for example `28` keeps exactly 28 components).
- Automatic thin PCA path for high-dimensional datasets where features > samples.
- Python bindings (`knn_core`) with a minimal, clean API.
- Build system based on CMake + scikit-build-core + pybind11.

## Repository Structure

```text
.
├── CMakeLists.txt
├── CMakePresets.json
├── pyproject.toml
├── include/
│   ├── KNN.hpp
│   ├── KNNEngine.hpp
│   └── PCA.hpp
├── src/
│   ├── KNN.cpp
│   ├── KNNEngine.cpp
│   ├── PCA.cpp
│   └── binding.cpp
└── examples/
    └── script.py
```

## How It Works

1. `KNNEngine.train(X, y, scale)` fits PCA on `X` and projects training data.
2. The projected training data is passed to the internal KNN classifier.
3. During inference, incoming samples are projected using the same PCA transform.
4. KNN predicts labels by majority vote among the nearest neighbors.

### PCA Modes

- Standard mode: used when number of features <= number of samples.
- Thin mode: used when number of features > number of samples (memory- and compute-friendly formulation).

## Requirements

- Python 3.8+
- CMake 3.18+
- C++17 compiler
- Ninja (recommended generator)

Python dependencies are declared in `pyproject.toml`:

- `numpy`
- `scikit-learn`

## Installation

### Option 1: Install from PyPI (recommended)

```bash
pip install knn-engine-core==0.1.0
```

This installs the published package and makes `knn_core` importable.

Quick verification after install:

```bash
python -c "import knn_core; cfg = knn_core.KNNConfig(); print('knn_core OK | k =', cfg.k, '| variance =', cfg.variance)"
```

### Option 2: Install from source (editable)

From the project root:

```bash
python -m pip install -U pip
python -m pip install -e .
```

### Option 3: Build with CMake presets

```bash
cmake --preset Release
cmake --build --preset Release
```

Artifacts are generated under `out/build/Release`.

## Quick Start

```python
import numpy as np
import knn_core

# Example data
X = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3],
    [5.9, 3.0, 5.1, 1.8],
], dtype=np.float64)

y = ["setosa", "setosa", "virginica", "virginica"]

cfg = knn_core.KNNConfig()
cfg.k = 3
cfg.variance = 0.95

engine = knn_core.KNNEngine(cfg)
engine.train(X, y, scale=False)

sample = np.array([6.0, 3.1, 5.0, 1.9], dtype=np.float64)
pred = engine.predict(sample)
print(pred)
```

## Python API

### `KNNConfig`

- `k: int` (default: `3`)
  - Number of neighbors used for voting.
- `variance: float` (default: `0.95`)
  - If `< 1.0`: treated as explained-variance threshold.
  - If `>= 1.0`: treated as fixed number of principal components.

### `KNNEngine(config: KNNConfig)`

- `train(X, y, scale=False)`
  - `X`: 2D `numpy.ndarray` (`float64` recommended), shape `(n_samples, n_features)`
  - `y`: list of labels (strings)
  - `scale`: whether to standardize features before PCA
- `predict(sample)`
  - `sample`: 1D feature vector
  - returns predicted label (`str`)
- `predict_batch(samples)`
  - `samples`: 2D feature matrix
  - returns list of predicted labels (`list[str]`)

## Benchmark Results

The following results were produced using the provided evaluation script pattern (`examples/script.py`) with train/test split and fixed random seed.

| Dataset | PCA Mode | Components | Accuracy |
|---|---|---:|---:|
| Olivetti Faces | Thin | 62 | 92.50% |
| Iris Flower | Standard | 3 | 100.00% |
| Handwritten Digits | Standard | 28 | 98.06% |

Console logs:

```text
--- Testing Olivetti Faces ---
[PCA] Fit complete. Mode: Thin | Components: 62
[Engine] Trained successfully. Reduced to 62 dimensions.
Accuracy: 92.50%

--- Testing Iris Flower ---
[PCA] Fit complete. Mode: Standard | Components: 3
[Engine] Trained successfully. Reduced to 3 dimensions.
Accuracy: 100.00%

--- Testing Handwritten Digits ---
[PCA] Fit complete. Mode: Standard | Components: 28
[Engine] Trained successfully. Reduced to 28 dimensions.
Accuracy: 98.06%
```

## Running the Example Script

```bash
python examples/script.py
```

## Notes and Limitations

- Labels are currently represented as strings in the C++ core API.
- Distance metric is Euclidean (squared Euclidean for ranking).
- Tie-breaking follows `std::map` key ordering when vote counts are equal.

## Development

Typical local workflow:

```bash
python -m pip install -e .
python examples/script.py
```

For release-style native builds:

```bash
cmake --preset Release
cmake --build --preset Release
```