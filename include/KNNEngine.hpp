#pragma once
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

#include "KNN.hpp"
#include "PCA.hpp"
#include "Parser.hpp"

// Move Config outside the class so its definition is fully "complete"
// before being used as a default argument in the KNNEngine constructor.
struct KNNConfig {
  int k = 3;
  double variance = 0.95;
  int label_col = -1;
};

class KNNEngine {
 public:
  explicit KNNEngine(KNNConfig config = KNNConfig());

  void train(const Eigen::MatrixXd& X, const std::vector<std::string>& y,
             bool scale = false);

  bool train_from_file(const std::string& csv_path, bool scale = false);
  std::string predict(const Eigen::VectorXd& raw_input) const;
  std::vector<std::string> predict_batch(const Eigen::MatrixXd& inputs) const;

 private:
  KNNConfig cfg;
  std::unique_ptr<PCA> pca;
  std::unique_ptr<KNN> knn;
};