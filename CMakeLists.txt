cmake_minimum_required(VERSION 3.5.0)
project(Deep_Library_CPP VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(Deep_Library_CPP src/main.cpp)
target_include_directories(Deep_Library_CPP PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(Deep_Library_CPP PUBLIC cxx_std_23)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
target_compile_options(Deep_Library_CPP PUBLIC -O3 -mavx2 -mfma -msse4.1 -msse4.2)
endif()

# Eigen 3.4 - fair fixed-size baseline for the benchmark.
# Vendored under third_party/Eigen if present (offline); otherwise fetched.
if(EXISTS ${CMAKE_SOURCE_DIR}/third_party/Eigen/Dense)
    set(EIGEN_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party)
    message(STATUS "Eigen: using vendored copy in third_party/")
else()
    include(FetchContent)
    FetchContent_Declare(eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
    FetchContent_MakeAvailable(eigen)
    set(EIGEN_INCLUDE_DIR ${eigen_SOURCE_DIR})
    message(STATUS "Eigen: fetched 3.4.0 via FetchContent")
endif()

add_executable(benchmark benchmarks/benchmark.cpp)
target_include_directories(benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN_INCLUDE_DIR})
target_compile_features(benchmark PUBLIC cxx_std_23)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
target_compile_options(benchmark PUBLIC -O3 -mavx2 -mfma -msse4.1 -msse4.2)
endif()