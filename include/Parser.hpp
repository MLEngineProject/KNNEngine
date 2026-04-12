#pragma once
#include <Eigen/Dense>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

class Parser {
 public:
  using MatrixRM =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  explicit Parser(fs::path filePath, int labelColumnIndex = -1);

  bool parse();

  const MatrixRM& getFeatures() const { return features; }
  const std::vector<std::string>& getLabels() const { return labels; }
  const std::vector<std::string>& getHeader() const { return columnNames; }

 private:
  fs::path path;
  int labelIdx;
  MatrixRM features;
  std::vector<std::string> labels;
  std::vector<std::string> columnNames;
};