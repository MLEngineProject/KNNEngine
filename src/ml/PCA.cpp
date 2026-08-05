#include "ml/PCA.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

namespace affineflow::ml {

PCA::PCA(double components) : threshold(components), num_components(0) {}

void PCA::fit(const Eigen::Ref<const RowMatrixXd>& X, bool scale) {
  if (X.rows() == 0 || X.cols() == 0) {
    throw std::invalid_argument("Cannot fit PCA on empty data array.");
  }
  if (!X.allFinite()) {
    throw std::invalid_argument("Input data contains NaN or Infinity values.");
  }
  mean = X.colwise().mean();
  RowMatrixXd centered = X.rowwise() - mean;

  if (scale) {
    std_dev =
        (centered.array().square().colwise().sum() / (X.rows() - 1)).sqrt();
    std_dev = std_dev.unaryExpr([](double v) { return v < 1e-9 ? 1.0 : v; });
    centered.array().rowwise() /= std_dev.array();
  } else {
    std_dev = Eigen::RowVectorXd::Ones(X.cols());
  }

  int N = X.rows();
  int D = X.cols();

  if (D > N) {
    Eigen::MatrixXd L = (centered * centered.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(L);

    Eigen::VectorXd evals = solver.eigenvalues().reverse();
    Eigen::MatrixXd evecs_L = solver.eigenvectors().rowwise().reverse();

    Eigen::MatrixXd evecs = centered.transpose() * evecs_L;

    for (int i = 0; i < evecs.cols(); ++i) {
      double norm = evecs.col(i).norm();
      if (norm > 1e-9) {
        evecs.col(i) /= norm;
      } else {
        evecs.col(i).setZero();
      }
    }
    apply_threshold(evals, evecs, true);
  } else {
    Eigen::MatrixXd cov = (centered.transpose() * centered) / double(N - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);

    Eigen::VectorXd evals = solver.eigenvalues().reverse();
    Eigen::MatrixXd evecs = solver.eigenvectors().rowwise().reverse();

    apply_threshold(evals, evecs, false);
  }
}

void PCA::apply_threshold(const Eigen::VectorXd& evals,
                          const Eigen::MatrixXd& evecs, bool is_thin_mode) {
  num_components = 0;

  if (threshold >= 1.0) {
    num_components =
        std::min(static_cast<int>(threshold), static_cast<int>(evals.size()));
  } else {
    double total_variance = evals.sum();
    double cumulative_variance = 0.0;
    for (int i = 0; i < evals.size(); ++i) {
      num_components++;
      cumulative_variance += evals[i];
      if ((cumulative_variance / total_variance) >= threshold) break;
    }
  }

  projection_matrix = evecs.leftCols(num_components);
}

RowMatrixXd PCA::transform(const Eigen::Ref<const RowMatrixXd>& X) const {
  if (X.cols() != mean.size()) {
    throw std::invalid_argument("Feature dimension mismatch. Expected " +
                                std::to_string(mean.size()));
  }
  if (!X.allFinite()) {
    throw std::invalid_argument("Input data contains NaN or Infinity values.");
  }
  RowMatrixXd centered = X.rowwise() - mean;
  RowMatrixXd standardized = centered.array().rowwise() / std_dev.array();
  return standardized * projection_matrix;
}

RowMatrixXd PCA::fit_transform(const Eigen::Ref<const RowMatrixXd>& X,
                               bool scale) {
  fit(X, scale);
  return transform(X);
}

}  // namespace affineflow::ml