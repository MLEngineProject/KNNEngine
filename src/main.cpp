#include <iostream>

#include "KNNEngine.hpp"

int main() {
  KNNConfig cfg;
  cfg.k = 5;
  cfg.variance = 0.98;

  KNNEngine engine(cfg);

  if (!engine.train_from_file("data/iris_train.data")) {
    return 1;
  }

  Eigen::VectorXd sample(4);
  sample << 5.1, 3.5, 1.4, 0.2;

  std::string result = engine.predict(sample);
  std::cout << "Prediction for custom sample: " << result << std::endl;

  return 0;
}