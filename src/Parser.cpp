#include "Parser.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

Parser::Parser(fs::path filePath, int labelColumnIndex)
    : path(std::move(filePath)), labelIdx(labelColumnIndex) {}

bool Parser::parse() {
  if (!fs::exists(path)) return false;

  std::ifstream file(path);
  std::string line;

  if (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string colName;
    while (std::getline(ss, colName, ',')) {
      columnNames.push_back(colName);
    }
  }

  if (labelIdx == -1) labelIdx = columnNames.size() - 1;

  std::vector<std::vector<double>> temp_data;
  while (std::getline(file, line)) {
    if (line.empty()) continue;

    std::stringstream ss(line);
    std::string value;
    std::vector<double> row_features;
    int current_col = 0;

    while (std::getline(ss, value, ',')) {
      if (current_col == labelIdx) {
        labels.push_back(value);
      } else {
        try {
          row_features.push_back(std::stod(value));
        } catch (...) {
          row_features.push_back(0.0); 
        }
      }
      current_col++;
    }
    temp_data.push_back(row_features);
  }

  int num_rows = temp_data.size();
  int num_features = columnNames.size() - 1;

  features.resize(num_rows, num_features);
  for (int i = 0; i < num_rows; ++i) {
    features.row(i) =
        Eigen::Map<Eigen::VectorXd>(temp_data[i].data(), num_features);
  }

  return true;
}