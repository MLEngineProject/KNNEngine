include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_executable(ml_tests 
    cpp/test_pca.cpp
    cpp/test_knn.cpp
)

target_link_libraries(ml_tests PRIVATE AffineFlow-ML::Core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(ml_tests)