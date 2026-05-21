#include "KNNEngine.hpp"

#include <iostream>

KNNEngine::KNNEngine(KNNConfig config) : cfg(config) {
  pca = std::make_unique<PCA>(cfg.variance);
  knn = std::make_unique<KNN>(cfg.k);
}

void KNNEngine::train(const Eigen::MatrixXd& X,
                      const std::vector<std::string>& y, bool scale) {
  pca->fit(X, scale);
  Eigen::MatrixXd x_reduced = pca->transform(X);

  knn->train(x_reduced, y);

  std::cout << "[Engine] Trained successfully. Reduced to "
            << pca->getComponentCount() << " dimensions." << std::endl;
}

std::string KNNEngine::predict(const Eigen::VectorXd& raw_input) const {
  Eigen::MatrixXd projected = pca->transform(raw_input.transpose());
  return knn->predict(projected.transpose());
}

std::vector<std::string> KNNEngine::predict_batch(
    const Eigen::MatrixXd& inputs) const {
  Eigen::MatrixXd projected = pca->transform(inputs);
  std::vector<std::string> predictions(projected.rows());

#pragma omp parallel for
  for (int i = 0; i < projected.rows(); ++i) {
    predictions[i] = knn->predict(projected.row(i).transpose());
  }
  return predictions;
}