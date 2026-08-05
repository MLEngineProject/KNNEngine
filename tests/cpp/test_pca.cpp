#include <gtest/gtest.h>

#include "ml/PCA.hpp"

using namespace affineflow::ml;

TEST(PCATest, StandardModeVariance) {
  // 4 samples, 3 features (N > D) -> Standard PCA
  RowMatrixXd X(4, 3);
  X << 1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0;

  PCA pca(0.99);  // Keep 99% variance
  pca.fit(X, false);

  // Since the features are perfectly correlated, 1 component explains 100% of
  // variance
  EXPECT_EQ(pca.getComponentCount(), 1);

  RowMatrixXd transformed = pca.transform(X);
  EXPECT_EQ(transformed.rows(), 4);
  EXPECT_EQ(transformed.cols(), 1);
}

TEST(PCATest, ThinModeFixedComponents) {
  // 3 samples, 5 features (D > N) -> Thin PCA
  RowMatrixXd X(3, 5);
  X << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
      0.0;

  PCA pca(2.0);  // Keep exactly 2 components
  pca.fit(X, false);

  EXPECT_EQ(pca.getComponentCount(), 2);

  RowMatrixXd transformed = pca.transform(X);
  EXPECT_EQ(transformed.rows(), 3);
  EXPECT_EQ(transformed.cols(), 2);
}