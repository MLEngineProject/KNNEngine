#include "KNN.hpp"

#include <algorithm>
#include <numeric>

KNN::KNN(int k) : k_neighbors(k) {}

void KNN::train(const Eigen::MatrixXd& training_data,
                const std::vector<std::string>& training_labels) {
  train_X = training_data;
  train_y = training_labels;
}

std::string KNN::predict(const Eigen::VectorXd& query_point) const {
  Eigen::VectorXd distances =
      (train_X.rowwise() - query_point.transpose()).rowwise().squaredNorm();

  std::vector<int> indices(train_X.rows());
  std::iota(indices.begin(), indices.end(), 0);

  std::nth_element(
      indices.begin(), indices.begin() + k_neighbors, indices.end(),
      [&distances](int i, int j) { return distances(i) < distances(j); });

  std::map<std::string, int> votes;
  for (int i = 0; i < k_neighbors; ++i) {
    votes[train_y[indices[i]]]++;
  }

  return std::max_element(
             votes.begin(), votes.end(),
             [](const auto& a, const auto& b) { return a.second < b.second; })
      ->first;
}