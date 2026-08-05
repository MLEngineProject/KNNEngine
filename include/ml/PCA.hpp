#pragma once
#include <Eigen/Dense>

namespace affineflow::ml {

using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class PCA {
 public:
  explicit PCA(double components = 0.95);

  void fit(const Eigen::Ref<const RowMatrixXd>& X, bool scale = false);
  RowMatrixXd transform(const Eigen::Ref<const RowMatrixXd>& X) const;
  RowMatrixXd fit_transform(const Eigen::Ref<const RowMatrixXd>& X,
                            bool scale = false);

  int getComponentCount() const { return num_components; }

  double get_threshold() const { return threshold; }
  int get_num_components() const { return num_components; }
  const Eigen::RowVectorXd& get_mean() const { return mean; }
  const Eigen::RowVectorXd& get_std_dev() const { return std_dev; }
  const Eigen::MatrixXd& get_projection_matrix() const {
    return projection_matrix;
  }

  void set_state(double thresh, int n_comp, const Eigen::RowVectorXd& m,
                 const Eigen::RowVectorXd& s, const Eigen::MatrixXd& proj) {
    threshold = thresh;
    num_components = n_comp;
    mean = m;
    std_dev = s;
    projection_matrix = proj;
  }

 private:
  double threshold;
  int num_components;
  Eigen::RowVectorXd mean;
  Eigen::RowVectorXd std_dev;
  Eigen::MatrixXd projection_matrix;

  void apply_threshold(const Eigen::VectorXd& evals,
                       const Eigen::MatrixXd& evecs, bool is_thin_mode);
};

}  // namespace affineflow::ml