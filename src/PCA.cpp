#include "PCA.hpp"

#include <iostream>

PCA::PCA(double components) : threshold(components), num_components(0) {}

void PCA::fit(const Eigen::MatrixXd& X) {
  // 1. Calculate Mean and Center
  mean = X.colwise().mean();
  Eigen::MatrixXd centered = X.rowwise() - mean;

  // 2. Transpose Trick (Matches your previous project speed)
  // Compute L = A * A^T (Samples x Samples)
  Eigen::MatrixXd L = (centered * centered.transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(L);
  Eigen::VectorXd evals = solver.eigenvalues().reverse();
  Eigen::MatrixXd evecs_L = solver.eigenvectors().rowwise().reverse();

  // 3. Lift to Pixel Space: V = A^T * U
  Eigen::MatrixXd evecs = centered.transpose() * evecs_L;

  // 4. CRITICAL: Normalize to ensure an orthonormal basis
  for (int i = 0; i < evecs.cols(); ++i) {
    evecs.col(i).normalize();
  }

  // 5. Select components for variance threshold
  double total_variance = evals.sum();
  double cumulative_variance = 0.0;
  num_components = 0;
  for (int i = 0; i < evals.size(); ++i) {
    num_components++;
    cumulative_variance += evals[i];
    if ((cumulative_variance / total_variance) >= threshold) break;
  }

  projection_matrix = evecs.leftCols(num_components);
  std::cout << "[PCA] Fast fit restored. Components: " << num_components
            << std::endl;
}

Eigen::MatrixXd PCA::transform(const Eigen::MatrixXd& X) const {
  Eigen::MatrixXd centered = X.rowwise() - mean;
  Eigen::MatrixXd standardized = centered.array().rowwise() / std_dev.array();
  return standardized * projection_matrix;
}

Eigen::MatrixXd PCA::fit_transform(const Eigen::MatrixXd& X) {
  fit(X);
  return transform(X);
}