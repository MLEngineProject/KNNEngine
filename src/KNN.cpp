#include "KNN.hpp"

#include <algorithm>
#include <numeric>

KNN::KNN(int k) : k_neighbors(k) {}

void KNN::train(const Eigen::MatrixXd& training_data,
                const std::vector<std::string>& training_labels) {
  train_X = training_data;
  train_y = training_labels;
  train_X_sq_norms = train_X.rowwise().squaredNorm();
}

std::string KNN::predict(const Eigen::VectorXd& query_point) const {
  if (train_X.rows() == 0 || k_neighbors <= 0) return "Unknown";

  Eigen::VectorXd distances = train_X_sq_norms.array() -
                              2 * (train_X * query_point).array() +
                              query_point.squaredNorm();

  std::vector<int> indices(train_X.rows());
  std::iota(indices.begin(), indices.end(), 0);

  int k_eff = std::min(k_neighbors, static_cast<int>(train_X.rows()));

  if (k_eff < train_X.rows()) {
    std::nth_element(
        indices.begin(), indices.begin() + k_eff, indices.end(),
        [&distances](int i, int j) { return distances(i) < distances(j); });
  }

  std::map<std::string, int> votes;
  for (int i = 0; i < k_eff; ++i) {
    votes[train_y[indices[i]]]++;
  }

  return std::max_element(
             votes.begin(), votes.end(),
             [](const auto& a, const auto& b) { return a.second < b.second; })
      ->first;
}