cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

project(KNN_ENGINE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()

FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.1.tar.gz
)
FetchContent_MakeAvailable(pybind11)

find_package(OpenMP REQUIRED)

pybind11_add_module(knn_core NO_EXTRAS
    src/binding.cpp 
    src/PCA.cpp 
    src/KNN.cpp
    src/KNNEngine.cpp
)

target_include_directories(knn_core SYSTEM PRIVATE 
    include 
    ${eigen_SOURCE_DIR}
)

target_link_libraries(knn_core PRIVATE pybind11::module OpenMP::OpenMP_CXX)

target_compile_definitions(knn_core PRIVATE EIGEN_USE_THREADS)

install(TARGETS knn_core DESTINATION .)