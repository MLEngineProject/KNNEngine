import knn_core
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import time

def run_face_recognition():
    cfg = knn_core.KNNConfig()
    cfg.k = 3
    cfg.variance = 0.98
    
    engine = knn_core.KNNEngine(cfg)

    print("--- Loading Olivetti Faces ---")
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data.astype(np.float64)
    y = faces.target.astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    print(f"Training on {len(X_train)} samples...")
    start_train = time.time()
    engine.train(X_train, y_train) 
    print(f"Training completed in {time.time() - start_train:.4f} seconds.")

    print(f"Predicting {len(X_test)} samples...")
    correct = 0
    start_pred = time.time()
    
    for i in range(len(X_test)):
        prediction = engine.predict(X_test[i])
        if prediction == y_test[i]:
            correct += 1

    total_time = time.time() - start_pred
    accuracy = (correct / len(X_test)) * 100

    print("\n" + "="*30)
    print("   EIGENFACES RESULTS")
    print("="*30)
    print(f"Accuracy:      {accuracy:.2f}%")
    print(f"Correct:       {correct}/{len(X_test)}")
    print(f"Avg Pred Time: {(total_time/len(X_test))*1000:.2f} ms/image")
    print("="*30)

if __name__ == "__main__":
    run_face_recognition()