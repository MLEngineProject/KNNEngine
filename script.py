import knn_core
import numpy as np
from sklearn.datasets import fetch_olivetti_faces, load_iris, load_digits
from sklearn.model_selection import train_test_split

def test_dataset(name, X, y, k, variance, scale):
    print(f"\n--- Testing {name} ---")
    cfg = knn_core.KNNConfig()
    cfg.k = k
    cfg.variance = variance
    engine = knn_core.KNNEngine(cfg)

    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float64), y.astype(str), test_size=0.2, stratify=y, random_state=42
    )

    engine.train(X_train, y_train.tolist(), scale)
    
    correct = 0
    for i in range(len(X_test)):
        if engine.predict(X_test[i]) == y_test[i]:
            correct += 1
    
    print(f"Accuracy: {correct/len(X_test)*100:.2f}%")

if __name__ == "__main__":
    faces = fetch_olivetti_faces()
    test_dataset("Olivetti Faces", faces.data, faces.target, k=1, variance=0.90, scale=False)

    iris = load_iris()
    test_dataset("Iris Flower", iris.data, iris.target, k=3, variance=0.99, scale=False)

    digits = load_digits()
    test_dataset("Handwritten Digits", digits.data, digits.target, k=3, variance=0.95, scale=False)