# AffineFlow-ML

[![PyPI version](https://img.shields.io/pypi/v/AffineFlow-ML?logo=pypi&logoColor=white)](https://pypi.org/project/AffineFlow-ML/)
[![Python](https://img.shields.io/pypi/pyversions/AffineFlow-ML?logo=python&logoColor=white)](https://pypi.org/project/AffineFlow-ML/)
[![Build system](https://img.shields.io/badge/build-scikit--build--core-blue?logo=cmake&logoColor=white)](https://scikit-build-core.readthedocs.io/)
[![Bindings](https://img.shields.io/badge/bindings-pybind11-4C72B0?logo=python&logoColor=white)](https://pybind11.readthedocs.io/)

A high-performance Machine Learning engine for the AffineFlow ecosystem. 

Designed for rapid experimentation without the Python Global Interpreter Lock (GIL) overhead, `AffineFlow-ML` executes compute-intensive tasks strictly in native C++ using Eigen and OpenBLAS. It features a dynamically compiled multi-architecture backend (Generic, AVX2, AVX-512) for maximum hardware-accelerated throughput safely dispatched at runtime.

## Highlights

- **Dynamic Hardware Dispatch**: Automatically detects your CPU at runtime to dispatch highly optimized AVX2 or AVX-512 instructions.
- **Zero-Copy Buffer Protocol**: Uses `Eigen::Ref` and RowMajor structures to natively ingest NumPy arrays without memory duplication.
- **Advanced KNN Classifier**: OpenMP parallelized C++17 implementation with efficient top-k neighbor selection (`std::nth_element`).
- **Distance Metrics**: Native vectorized support for Euclidean, Manhattan, and Cosine distance algorithms.
- **Weighted Voting**: Choose between standard `uniform` voting or `distance`-weighted predictions.
- **Adaptive PCA**: Built-in dimensionality reduction with standard and thin-SVD modes for high-dimensional data.

## Installation

Install the released wheel from PyPI:

    pip install AffineFlow-ML

## Quick Start

```py
import numpy as np
import affineflow_ml as ml

# Example data
X_train = np.random.rand(1000, 50).astype(np.float64)
y_train = ["class_A" if x[0] > 0.5 else "class_B" for x in X_train]

# Configure the engine with new features
cfg = ml.KNNConfig()
cfg.k = 5
cfg.variance = 0.95
cfg.metric = ml.DistanceMetric.COSINE
cfg.weights = ml.Weights.DISTANCE

engine = ml.KNNEngine(cfg)
engine.train(X_train, y_train, scale=True)

X_test = np.random.rand(10, 50).astype(np.float64)
predictions = engine.predict_batch(X_test)
print(predictions)
```

## Performance Benchmark

`AffineFlow-ML` achieves strict mathematical parity with Scikit-Learn while delivering up to **5.4x faster inference and transformation pipelines** on high-dimensional datasets. Powered by a multithreaded C++17 Eigen backend with dynamic hardware dispatch (AVX2/AVX-512) and OpenMP parallelism, it eliminates Python loop and GIL bottlenecks.

| Dataset | Metric & Weights | Scikit-Learn Time | AffineFlow-ML Time | Speedup | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Olivetti Faces** (400 samples, 4096 dims $\rightarrow$ PCA 90%) | Cosine / Distance | 0.4134s | **0.0760s** | **5.44x** | 90.00% |
| **Handwritten Digits** (1797 samples, 64 dims $\rightarrow$ PCA 95%) | Manhattan / Uniform | 0.0808s | **0.0354s** | **2.29x** | 98.61% |
| **Iris Flower** (150 samples, 4 dims $\rightarrow$ PCA 99%) | Euclidean / Uniform | 0.0105s | **0.0055s** | **1.91x** | 100.00% |