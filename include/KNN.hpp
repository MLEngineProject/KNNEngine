#pragma once
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

class KNN {
 public:
  explicit KNN(int k = 3);

  void train(const Eigen::MatrixXd& training_data,
             const std::vector<std::string>& training_labels);

  // Predict using vectorized distance calculations
  std::string predict(const Eigen::VectorXd& query_point) const;

 private:
  int k_neighbors;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      train_X;
  std::vector<std::string> train_y;
};