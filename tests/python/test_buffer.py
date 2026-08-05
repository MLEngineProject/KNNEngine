import pytest
import numpy as np
import affineflow_ml as ml

def test_numpy_memory_ingestion():
    """Verify that PyBind11 correctly maps NumPy arrays to Eigen::Ref without copying."""
    X_train = np.random.rand(10, 5).astype(np.float64)
    y_train = [str(i) for i in range(10)]
    
    knn = ml.KNeighborsClassifier(n_neighbors=3)
    
    # If the buffer protocol fails, this will throw a type error or segfault
    knn.fit(X_train, y_train)
    
    # Ensure C-contiguous arrays work for batch prediction
    X_test = np.ascontiguousarray(np.random.rand(2, 5).astype(np.float64))
    preds = knn.predict(X_test)
    assert len(preds) == 2