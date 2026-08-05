#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ml/KNeighborsClassifier.hpp"
#include "ml/PCA.hpp"

namespace py = pybind11;
using namespace affineflow::ml;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(BACKEND_NAME, m) {
  m.doc() = "C++ Core Backend Engine for AffineFlow-ML";

  py::enum_<DistanceMetric>(m, "DistanceMetric")
      .value("EUCLIDEAN", DistanceMetric::EUCLIDEAN)
      .value("MANHATTAN", DistanceMetric::MANHATTAN)
      .value("COSINE", DistanceMetric::COSINE)
      .export_values();

  py::enum_<Weights>(m, "Weights")
      .value("UNIFORM", Weights::UNIFORM)
      .value("DISTANCE", Weights::DISTANCE)
      .export_values();

  py::class_<PCA>(m, "PCA", "Principal Component Analysis (PCA)")
      .def(py::init<double>(), py::arg("n_components") = 0.95)
      .def("fit", &PCA::fit, py::arg("X"), py::arg("scale") = false,
           py::call_guard<py::gil_scoped_release>())
      .def("transform", &PCA::transform, py::arg("X"),
           py::call_guard<py::gil_scoped_release>())
      .def("fit_transform", &PCA::fit_transform, py::arg("X"),
           py::arg("scale") = false, py::call_guard<py::gil_scoped_release>())
      .def(py::pickle(
          [](const PCA& p) {  // __getstate__
            return py::make_tuple(p.get_threshold(), p.get_num_components(),
                                  p.get_mean(), p.get_std_dev(),
                                  p.get_projection_matrix());
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 5) throw std::runtime_error("Invalid PCA state!");
            PCA p(t[0].cast<double>());
            p.set_state(t[0].cast<double>(), t[1].cast<int>(),
                        t[2].cast<Eigen::RowVectorXd>(),
                        t[3].cast<Eigen::RowVectorXd>(),
                        t[4].cast<Eigen::MatrixXd>());
            return p;
          }));

  py::class_<KNeighborsClassifier>(m, "KNeighborsClassifier",
                                   "K-Nearest Neighbors Classifier")
      .def(py::init<int, DistanceMetric, Weights>(), py::arg("n_neighbors") = 3,
           py::arg("metric") = DistanceMetric::EUCLIDEAN,
           py::arg("weights") = Weights::UNIFORM)
      .def("fit", &KNeighborsClassifier::fit, py::arg("X"), py::arg("y"),
           py::call_guard<py::gil_scoped_release>())
      .def("predict", &KNeighborsClassifier::predict, py::arg("X"),
           py::call_guard<py::gil_scoped_release>())
      .def(py::pickle(
          [](const KNeighborsClassifier& knn) {  // __getstate__
            return py::make_tuple(knn.get_k(), knn.get_metric(),
                                  knn.get_weights(), knn.get_n_features(),
                                  knn.get_train_X(), knn.get_train_y(),
                                  knn.get_train_X_norms());
          },
          [](py::tuple t) {  // __setstate__
            if (t.size() != 7) throw std::runtime_error("Invalid KNN state!");
            KNeighborsClassifier knn(t[0].cast<int>(),
                                     t[1].cast<DistanceMetric>(),
                                     t[2].cast<Weights>());
            knn.set_state(t[0].cast<int>(), t[1].cast<DistanceMetric>(),
                          t[2].cast<Weights>(), t[3].cast<int>(),
                          t[4].cast<RowMatrixXd>(),
                          t[5].cast<std::vector<std::string>>(),
                          t[6].cast<Eigen::VectorXd>());
            return knn;
          }));
}