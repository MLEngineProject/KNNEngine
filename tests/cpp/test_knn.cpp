#include <gtest/gtest.h>

#include "ml/KNeighborsClassifier.hpp"

using namespace affineflow::ml;

class KNNTest : public ::testing::Test {
 protected:
  RowMatrixXd train_X;
  std::vector<std::string> train_y;

  void SetUp() override {
    train_X.resize(4, 2);
    train_X << 0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0;
    train_y = {"A", "A", "B", "B"};
  }
};

TEST_F(KNNTest, EuclideanPrediction) {
  KNeighborsClassifier knn(1, DistanceMetric::EUCLIDEAN, Weights::UNIFORM);
  knn.fit(train_X, train_y);

  Eigen::VectorXd query(2);
  query << 0.5, 0.5;
  EXPECT_EQ(knn.predict_single(query), "A");

  query << 10.5, 10.5;
  EXPECT_EQ(knn.predict_single(query), "B");
}

TEST_F(KNNTest, ManhattanPrediction) {
  KNeighborsClassifier knn(3, DistanceMetric::MANHATTAN, Weights::UNIFORM);
  knn.fit(train_X, train_y);

  Eigen::VectorXd query(2);
  query << 2.0, 2.0;
  // Manhattan distances:
  // [0,0] -> 4.0
  // [1,1] -> 2.0
  // [10,10] -> 16.0
  // [11,11] -> 18.0
  // Nearest 3: A, A, B. Majority: A.
  EXPECT_EQ(knn.predict_single(query), "A");
}

TEST_F(KNNTest, DistanceWeightedVoting) {
  // Edge case: A tie in neighbors, but distance weighting breaks the tie
  train_X.resize(3, 2);
  train_X << 0.0, 0.0, 2.0, 2.0, 2.1, 2.1;
  train_y = {"A", "B", "B"};

  // K=3 will pull in 1 'A' and 2 'B's.
  // With uniform weights, 'B' wins (2 vs 1).
  KNeighborsClassifier knn_uniform(3, DistanceMetric::EUCLIDEAN,
                                   Weights::UNIFORM);
  knn_uniform.fit(train_X, train_y);

  Eigen::VectorXd query(2);
  query << 0.1, 0.1;
  EXPECT_EQ(knn_uniform.predict_single(query), "B");

  // With distance weights, the 'A' at [0,0] is so close that its weight
  // dominates
  KNeighborsClassifier knn_weighted(3, DistanceMetric::EUCLIDEAN,
                                    Weights::DISTANCE);
  knn_weighted.fit(train_X, train_y);
  EXPECT_EQ(knn_weighted.predict_single(query), "A");
}