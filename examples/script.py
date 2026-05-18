import knn_core
import numpy as np
import time
from sklearn.datasets import fetch_olivetti_faces, load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def test_dataset(name, X, y, k, variance, scale):
    print(f"\n--- Testing {name} ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float64), y.astype(str), test_size=0.2, stratify=y, random_state=42
    )

    # --- KNNEngine ---
    cfg = knn_core.KNNConfig()
    cfg.k = k
    cfg.variance = variance
    engine = knn_core.KNNEngine(cfg)

    t0 = time.perf_counter()
    engine.train(X_train, y_train.tolist(), scale)
    correct = sum(
        engine.predict(X_test[i]) == y_test[i]
        for i in range(len(X_test))
    )
    t1 = time.perf_counter()

    knn_time = t1 - t0
    knn_acc = correct / len(X_test) * 100
    print(f"KNNEngine  — Accuracy: {knn_acc:.2f}%  |  Time: {knn_time:.4f}s")

    # --- Sklearn Equivalent ---
    n_components = knn_core.KNNEngine(cfg)
    pca = PCA(n_components=variance)
    
    t0 = time.perf_counter()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    sklearn_knn = KNeighborsClassifier(n_neighbors=k)
    sklearn_knn.fit(X_train_pca, y_train)
    sklearn_preds = sklearn_knn.predict(X_test_pca)
    t1 = time.perf_counter()

    sk_time = t1 - t0
    sk_acc = accuracy_score(y_test, sklearn_preds) * 100
    print(f"Sklearn    — Accuracy: {sk_acc:.2f}%  |  Time: {sk_time:.4f}s")
    print(f"Speedup: {sk_time / knn_time:.2f}x")

if __name__ == "__main__":
    faces = fetch_olivetti_faces()
    test_dataset("Olivetti Faces", faces.data, faces.target, k=1, variance=0.90, scale=False)
    iris = load_iris()
    test_dataset("Iris Flower", iris.data, iris.target, k=3, variance=0.99, scale=False)
    digits = load_digits()
    test_dataset("Handwritten Digits", digits.data, digits.target, k=3, variance=0.95, scale=False)
