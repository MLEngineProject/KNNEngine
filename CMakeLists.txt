cmake_minimum_required(VERSION 3.18)

project(KNN_ENGINE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_TESTING OFF CACHE INTERNAL "Disable tests globally")
set(EIGEN_BUILD_TESTING OFF CACHE INTERNAL "Disable Eigen tests")
set(EIGEN_BUILD_DOC OFF CACHE INTERNAL "Disable Eigen docs")

find_package(pybind11 CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(eigen)

pybind11_add_module(knn_core 
    src/binding.cpp 
    src/PCA.cpp 
    src/KNN.cpp
    src/KNNEngine.cpp
)

target_link_libraries(knn_core PRIVATE pybind11::module Eigen3::Eigen)

install(TARGETS knn_core DESTINATION .)