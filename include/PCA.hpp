#pragma once
#include <Eigen/Dense>

class PCA {
 public:
  // components > 1: fixed count. components < 1: variance threshold (e.g. 0.95)
  explicit PCA(double components = 0.95);

  void fit(const Eigen::MatrixXd& X);
  Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const;
  Eigen::MatrixXd fit_transform(const Eigen::MatrixXd& X);

  int getComponentCount() const { return num_components; }

 private:
  double threshold;
  int num_components;
  Eigen::RowVectorXd mean;
  Eigen::RowVectorXd std_dev;
  Eigen::MatrixXd projection_matrix;
};