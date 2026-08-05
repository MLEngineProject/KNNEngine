#pragma once
#include <Eigen/Dense>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace affineflow::ml {

using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

enum class DistanceMetric { EUCLIDEAN, MANHATTAN, COSINE };
enum class Weights { UNIFORM, DISTANCE };

class KNeighborsClassifier {
 public:
  explicit KNeighborsClassifier(
      int k = 3, DistanceMetric metric = DistanceMetric::EUCLIDEAN,
      Weights weights = Weights::UNIFORM);

  void fit(const Eigen::Ref<const RowMatrixXd>& X,
           const std::vector<std::string>& y);

  std::string predict_single(const Eigen::Ref<const Eigen::VectorXd>& x) const;
  std::vector<std::string> predict(
      const Eigen::Ref<const RowMatrixXd>& X) const;

  int get_k() const { return k_neighbors; }
  DistanceMetric get_metric() const { return metric_; }
  Weights get_weights() const { return weights_; }
  int get_n_features() const { return n_features_in_; }

  const RowMatrixXd& get_train_X() const { return train_X; }
  const std::vector<std::string>& get_train_y() const { return train_y; }
  const Eigen::VectorXd& get_train_X_norms() const { return train_X_norms; }

  void set_state(int k, DistanceMetric m, Weights w, int n_feat,
                 const RowMatrixXd& X, const std::vector<std::string>& y,
                 const Eigen::VectorXd& norms) {
    k_neighbors = k;
    metric_ = m;
    weights_ = w;
    n_features_in_ = n_feat;
    train_X = X;
    train_y = y;
    train_X_norms = norms;
  }

 private:
  int k_neighbors;
  DistanceMetric metric_;
  Weights weights_;
  int n_features_in_;  // For data validation

  RowMatrixXd train_X;
  std::vector<std::string> train_y;
  Eigen::VectorXd train_X_norms;
};

}  // namespace affineflow::ml