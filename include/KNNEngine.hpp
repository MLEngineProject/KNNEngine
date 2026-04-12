#pragma once
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

#include "KNN.hpp"
#include "PCA.hpp"

struct KNNConfig {
  int k = 3;
  double variance = 0.95;
};

class KNNEngine {
 public:
  explicit KNNEngine(KNNConfig config = KNNConfig());

  void train(const Eigen::MatrixXd& X, const std::vector<std::string>& y,
             bool scale = false);

  std::string predict(const Eigen::VectorXd& raw_input) const;
  std::vector<std::string> predict_batch(const Eigen::MatrixXd& inputs) const;

 private:
  KNNConfig cfg;
  std::unique_ptr<PCA> pca;
  std::unique_ptr<KNN> knn;
};