#include "KNNEngine.hpp"

#include <iostream>

KNNEngine::KNNEngine(KNNConfig config) : cfg(config) {
  pca = std::make_unique<PCA>(cfg.variance);
  knn = std::make_unique<KNN>(cfg.k);
}

void KNNEngine::train(const Eigen::MatrixXd& X,
                      const std::vector<std::string>& y) {
  Eigen::MatrixXd x_reduced = pca->fit_transform(X);
  knn->train(x_reduced, y);

  std::cout << "[Engine] Trained directly on memory. Dimensions: "
            << pca->getComponentCount() << std::endl;
}

bool KNNEngine::train_from_file(const std::string& csv_path) {
  Parser parser(csv_path, cfg.label_col);
  if (!parser.parse()) return false;

  Eigen::MatrixXd x_reduced = pca->fit_transform(parser.getFeatures());

  knn->train(x_reduced, parser.getLabels());

  std::cout << "[Engine] Trained successfully on " << x_reduced.rows()
            << " samples. Reduced to " << pca->getComponentCount()
            << " dimensions." << std::endl;
  return true;
}

std::string KNNEngine::predict(const Eigen::VectorXd& raw_input) const {
  Eigen::MatrixXd input_mat = raw_input.transpose();
  Eigen::VectorXd projected = pca->transform(input_mat).transpose();
  return knn->predict(projected);
}

std::vector<std::string> KNNEngine::predict_batch(
    const Eigen::MatrixXd& inputs) const {
  Eigen::MatrixXd projected = pca->transform(inputs);
  std::vector<std::string> predictions;
  for (int i = 0; i < projected.rows(); ++i) {
    predictions.push_back(knn->predict(projected.row(i)));
  }
  return predictions;
}