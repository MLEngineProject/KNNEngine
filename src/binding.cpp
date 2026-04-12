#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "KNNEngine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(knn_core, m) {
  m.doc() = "C++ KNN Engine with Adaptive PCA support";

  py::class_<KNNConfig>(m, "KNNConfig")
      .def(py::init<>())
      .def_readwrite("k", &KNNConfig::k)
      .def_readwrite("variance", &KNNConfig::variance)
      .def_readwrite("label_col", &KNNConfig::label_col);

  py::class_<KNNEngine>(m, "KNNEngine")
      .def(py::init<KNNConfig>())
      .def("train", &KNNEngine::train, py::arg("X"), py::arg("y"),
           py::arg("scale") = false)
      .def("train_from_file", &KNNEngine::train_from_file, py::arg("csv_path"),
           py::arg("scale") = false)
      .def("predict", &KNNEngine::predict)
      .def("predict_batch", &KNNEngine::predict_batch);
}