#include "ml/KNeighborsClassifier.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace affineflow::ml {

KNeighborsClassifier::KNeighborsClassifier(int k, DistanceMetric metric,
                                           Weights weights)
    : k_neighbors(k), metric_(metric), weights_(weights), n_features_in_(0) {}

void KNeighborsClassifier::fit(const Eigen::Ref<const RowMatrixXd>& X,
                               const std::vector<std::string>& y) {
  if (X.rows() != y.size()) {
    throw std::invalid_argument(
        "Number of samples in X does not match number of labels in y.");
  }

  // Explicit check for non-finite values (NaN, Inf)
  for (int i = 0; i < X.rows(); ++i) {
    for (int j = 0; j < X.cols(); ++j) {
      if (!std::isfinite(X(i, j))) {
        throw std::invalid_argument(
            "Input data contains NaN or Infinity values.");
      }
    }
  }

  n_features_in_ = X.cols();
  train_X = X;
  train_y = y;

  if (metric_ == DistanceMetric::EUCLIDEAN) {
    train_X_norms = train_X.rowwise().squaredNorm();
  } else if (metric_ == DistanceMetric::COSINE) {
    train_X_norms = train_X.rowwise().norm();
  }
}

std::string KNeighborsClassifier::predict_single(
    const Eigen::Ref<const Eigen::VectorXd>& x) const {
  if (train_X.rows() == 0 || k_neighbors <= 0) {
    throw std::runtime_error(
        "Estimator not fitted. Call 'fit' before 'predict'.");
  }
  if (x.size() != n_features_in_) {
    throw std::invalid_argument("Feature dimension mismatch. Expected " +
                                std::to_string(n_features_in_));
  }

  Eigen::VectorXd distances(train_X.rows());

  if (metric_ == DistanceMetric::EUCLIDEAN) {
    distances =
        train_X_norms.array() - 2.0 * (train_X * x).array() + x.squaredNorm();
  } else if (metric_ == DistanceMetric::MANHATTAN) {
    distances = (train_X.rowwise() - x.transpose()).cwiseAbs().rowwise().sum();
  } else if (metric_ == DistanceMetric::COSINE) {
    double q_norm = x.norm();
    Eigen::VectorXd dots = train_X * x;
    distances = 1.0 - (dots.array() / (train_X_norms.array() * q_norm + 1e-9));
  }

  std::vector<int> indices(train_X.rows());
  std::iota(indices.begin(), indices.end(), 0);

  int k_eff = std::min(k_neighbors, static_cast<int>(train_X.rows()));

  std::nth_element(
      indices.begin(), indices.begin() + k_eff, indices.end(),
      [&distances](int i, int j) { return distances(i) < distances(j); });

  std::map<std::string, double> votes;
  for (int i = 0; i < k_eff; ++i) {
    double weight = 1.0;
    if (weights_ == Weights::DISTANCE) {
      double d = std::max(distances(indices[i]), 1e-9);
      weight = 1.0 / d;
    }
    votes[train_y[indices[i]]] += weight;
  }

  return std::max_element(
             votes.begin(), votes.end(),
             [](const auto& a, const auto& b) { return a.second < b.second; })
      ->first;
}

std::vector<std::string> KNeighborsClassifier::predict(
    const Eigen::Ref<const RowMatrixXd>& X) const {
  if (X.cols() != n_features_in_) {
    throw std::invalid_argument(
        "Feature dimension mismatch in batch. Expected " +
        std::to_string(n_features_in_));
  }

  for (int i = 0; i < X.rows(); ++i) {
    for (int j = 0; j < X.cols(); ++j) {
      if (!std::isfinite(X(i, j))) {
        throw std::invalid_argument(
            "Input data contains NaN or Infinity values.");
      }
    }
  }

  std::vector<std::string> predictions(X.rows());

#pragma omp parallel for
  for (int i = 0; i < X.rows(); ++i) {
    predictions[i] = predict_single(X.row(i));
  }
  return predictions;
}

}  // namespace affineflow::ml