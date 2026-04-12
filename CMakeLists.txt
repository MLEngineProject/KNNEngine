cmake_minimum_required(VERSION 3.18)

# Disable the noisy FetchContent output
set(FETCHCONTENT_QUIET ON)

project(KNN_ENGINE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# This silences warnings coming from inside the included dependencies
set(CMAKE_MESSAGE_LOG_LEVEL WARNING)

include(FetchContent)

# 1. Declare Eigen
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)

# 2. Declare pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.1
    GIT_SHALLOW TRUE
)

# 3. Populate
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    # Adding as SYSTEM ignores warnings from the pybind11 headers
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# 4. Define the Python Module
pybind11_add_module(knn_core 
    src/binding.cpp 
    src/PCA.cpp 
    src/KNN.cpp
    src/KNNEngine.cpp
)

# 5. Configuration & Linking
target_include_directories(knn_core SYSTEM PRIVATE 
    include 
    ${eigen_SOURCE_DIR}
)

target_link_libraries(knn_core PRIVATE pybind11::module)

install(TARGETS knn_core DESTINATION .)