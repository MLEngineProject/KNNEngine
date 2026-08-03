cmake_minimum_required(VERSION 4.4.0)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

project(KNN_ENGINE LANGUAGES CXX)

# --- Global Compiler & System Settings ---
if(SKBUILD_FREE_THREADED)
    add_compile_definitions(Py_GIL_DISABLED=1)
    message(STATUS "Free-threaded Python detected: defining Py_GIL_DISABLED=1")
endif()

find_program(CCACHE_PROGRAM NAMES sccache ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "Compiler launcher" FORCE)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "Compiler launcher" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DEV_MODE "Use locally installed system libs for fast builds" OFF)

if(EMSCRIPTEN)
    # WebAssembly safe flags
    add_compile_options(-O3)
else()
    # Native Hardware Flags
    if(MSVC)
        add_compile_options(/MP)
    elseif(APPLE)
        add_compile_options(-O3 -ffast-math)
        list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
    else()
        add_compile_options(-O3 -march=native -ffast-math)
    endif()
endif()

# --- Dependencies ---
if(NOT EMSCRIPTEN)
    find_package(OpenMP)
endif()

if(DEV_MODE)
    find_package(Eigen3 REQUIRED)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 REQUIRED)
else()
    set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.cmake_deps" CACHE PATH "Cache fetch content" FORCE)
    include(FetchContent)

    # Fetch Eigen (Version bumped to match NNEngine)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    
    FetchContent_Declare(
        eigen 
        URL "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz" 
        SOURCE_SUBDIR ""
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(eigen)

    # Fetch Pybind11 (Version bumped to match NNEngine)
    if(EMSCRIPTEN)
        set(PYBIND11_FINDPYTHON ON)
        find_package(pybind11 REQUIRED)
    else()
        FetchContent_Declare(
            pybind11 
            URL "https://github.com/pybind/pybind11/archive/refs/tags/v3.0.4.tar.gz"
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# --- Target Definition ---
pybind11_add_module(knn_core NO_EXTRAS
    src/binding.cpp 
    src/PCA.cpp 
    src/KNN.cpp
    src/KNNEngine.cpp
)

# --- Include & Linking ---
if(DEV_MODE)
    target_link_libraries(knn_core PRIVATE Eigen3::Eigen)
else()
    target_include_directories(knn_core SYSTEM PRIVATE ${eigen_SOURCE_DIR})
endif()

target_include_directories(knn_core SYSTEM PRIVATE include)
target_link_libraries(knn_core PRIVATE pybind11::module)

# GUARD RAIL: Only link OpenMP if it was found AND we are not in Emscripten
if(OpenMP_CXX_FOUND AND NOT EMSCRIPTEN)
    target_link_libraries(knn_core PRIVATE OpenMP::OpenMP_CXX)
    if(MSVC)
        target_compile_options(knn_core PRIVATE /openmp:experimental)
    endif()
    message(STATUS "OpenMP found: Multi-threading enabled.")
else()
    message(WARNING "OpenMP NOT found or Emscripten detected: Falling back to single-threaded mode.")
endif()

target_compile_definitions(knn_core PRIVATE EIGEN_USE_THREADS)

install(TARGETS knn_core DESTINATION .)