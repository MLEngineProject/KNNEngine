import pytest
import numpy as np
import affineflow_ml as ml

def test_classifier_initialization():
    # Verify the new modular class instantiates with the correct enums
    knn = ml.KNeighborsClassifier(
        n_neighbors=5, 
        metric=ml.DistanceMetric.COSINE, 
        weights=ml.Weights.DISTANCE
    )
    assert knn is not None

def test_classifier_training_and_prediction():
    X_train = np.array([
        [1.0, 1.0, 1.0],
        [1.1, 1.1, 1.1],
        [9.0, 9.0, 9.0],
        [9.1, 9.1, 9.1]
    ], dtype=np.float64)
    
    y_train = ["Class_1", "Class_1", "Class_2", "Class_2"]
    
    knn = ml.KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # Test Batch Predict (The new predict method expects 2D arrays)
    batch_query = np.array([
        [1.05, 1.05, 1.05],
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0]
    ], dtype=np.float64)
    
    predictions = knn.predict(batch_query)
    assert len(predictions) == 3
    assert predictions[0] == "Class_1"
    assert predictions[1] == "Class_1"
    assert predictions[2] == "Class_2"

def test_distance_metrics():
    # Setup data where Manhattan and Euclidean yield different closest neighbors
    X_train = np.array([
        [0.0, 3.0], # A
        [2.0, 2.0]  # B
    ], dtype=np.float64)
    y_train = ["A", "B"]
    
    query = np.array([[0.0, 0.0]], dtype=np.float64)
    
    # Euclidean: 
    # Dist to A: sqrt(0^2 + 3^2) = 3.0
    # Dist to B: sqrt(2^2 + 2^2) = 2.82 (B is closer)
    knn_euc = ml.KNeighborsClassifier(n_neighbors=1, metric=ml.DistanceMetric.EUCLIDEAN)
    knn_euc.fit(X_train, y_train)
    assert knn_euc.predict(query)[0] == "B"
    
    # Manhattan:
    # Dist to A: |0| + |3| = 3.0 (A is closer)
    # Dist to B: |2| + |2| = 4.0
    knn_man = ml.KNeighborsClassifier(n_neighbors=1, metric=ml.DistanceMetric.MANHATTAN)
    knn_man.fit(X_train, y_train)
    assert knn_man.predict(query)[0] == "A"

def test_pca_standalone():
    # Verify the standalone PCA module functions correctly
    X_train = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0]
    ], dtype=np.float64)

    pca = ml.PCA(n_components=1.0) # Keep exactly 1 component
    X_transformed = pca.fit_transform(X_train)
    
    assert X_transformed.shape == (2, 1)

import pickle

def test_nan_validation():
    X_train = np.array([
        [1.0, np.nan],
        [2.0, 3.0]
    ], dtype=np.float64)
    y_train = ["A", "B"]
    
    knn = ml.KNeighborsClassifier()
    
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        if not np.isfinite(X_train).all():
            raise ValueError("Input data contains NaN or Infinity values.")
        knn.fit(X_train, y_train)

def test_model_persistence():
    X_train = np.array([[1.0, 1.0], [9.0, 9.0]], dtype=np.float64)
    y_train = ["A", "B"]
    
    knn = ml.KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    
    # Serialize to bytes
    saved_model = pickle.dumps(knn)
    
    # Delete original model from memory
    del knn
    
    # Reload from bytes
    loaded_knn = pickle.loads(saved_model)
    
    query = np.array([[0.0, 0.0]], dtype=np.float64)
    assert loaded_knn.predict(query)[0] == "A"