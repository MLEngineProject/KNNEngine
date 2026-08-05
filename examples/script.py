import affineflow_ml as ml
import numpy as np
import time
from sklearn.datasets import fetch_olivetti_faces, load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import accuracy_score

def test_dataset(name, X, y, k, variance, metric=ml.DistanceMetric.EUCLIDEAN, weights=ml.Weights.UNIFORM):
    print(f"\n--- Testing {name} ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float64), y.astype(str), test_size=0.2, stratify=y, random_state=42
    )

    # ==========================================
    # 1. AffineFlow-ML Pipeline
    # ==========================================
    t0 = time.perf_counter()
    
    # Standalone PCA
    pca = ml.PCA(n_components=variance)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Standalone KNN
    knn = ml.KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
    knn.fit(X_train_pca, y_train.tolist())
    
    # Predict
    preds = knn.predict(X_test_pca)
    
    t1 = time.perf_counter()

    correct = sum(p == y for p, y in zip(preds, y_test))
    af_time = t1 - t0
    af_acc = correct / len(X_test) * 100
    print(f"AffineFlow-ML  — Accuracy: {af_acc:.2f}%  |  Time: {af_time:.4f}s")

    # ==========================================
    # 2. Scikit-Learn Pipeline
    # ==========================================
    # Map AffineFlow enums to Sklearn string arguments
    sk_metric = 'euclidean'
    if metric == ml.DistanceMetric.MANHATTAN:
        sk_metric = 'manhattan'
    elif metric == ml.DistanceMetric.COSINE:
        sk_metric = 'cosine'
        
    sk_weights = 'uniform' if weights == ml.Weights.UNIFORM else 'distance'

    t0_sk = time.perf_counter()
    
    # Sklearn PCA (requires int for exact components, float for variance)
    n_comp = int(variance) if variance >= 1.0 else variance
    sk_pca = SklearnPCA(n_components=n_comp)
    X_train_sk_pca = sk_pca.fit_transform(X_train)
    X_test_sk_pca = sk_pca.transform(X_test)
    
    # Sklearn KNN
    sk_knn = SklearnKNN(n_neighbors=k, metric=sk_metric, weights=sk_weights)
    sk_knn.fit(X_train_sk_pca, y_train)
    
    # Predict
    sk_preds = sk_knn.predict(X_test_sk_pca)
    
    t1_sk = time.perf_counter()

    sk_time = t1_sk - t0_sk
    sk_acc = accuracy_score(y_test, sk_preds) * 100
    print(f"Scikit-Learn   — Accuracy: {sk_acc:.2f}%  |  Time: {sk_time:.4f}s")
    print(f"Speedup        — {sk_time / af_time:.2f}x")

if __name__ == "__main__":
    faces = fetch_olivetti_faces()
    test_dataset("Olivetti Faces (Cosine & Distance Weighted)", 
                 faces.data, faces.target, k=3, variance=0.90,
                 metric=ml.DistanceMetric.COSINE, weights=ml.Weights.DISTANCE)
    
    iris = load_iris()
    test_dataset("Iris Flower (Standard Euclidean)", 
                 iris.data, iris.target, k=3, variance=0.99)
                 
    digits = load_digits()
    test_dataset("Handwritten Digits (Manhattan)", 
                 digits.data, digits.target, k=3, variance=0.95,
                 metric=ml.DistanceMetric.MANHATTAN)