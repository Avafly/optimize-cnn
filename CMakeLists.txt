cmake_minimum_required(VERSION 3.20)
project(Optimize-CNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# libs path
set(LIB_ROOT "$ENV{HOME}/Documents/libs")

# openmp
find_package(OpenMP REQUIRED)
# openblas
set(OPENBLAS_INCLUDE_DIR "${LIB_ROOT}/openblas/include")
set(OPENBLAS_LIB_DIR "${LIB_ROOT}/openblas/lib")
# ncnn
if(NOT DEFINED ncnn_DIR)
    set(ncnn_DIR "${LIB_ROOT}/ncnn/lib/cmake/ncnn")
endif()
find_package(ncnn REQUIRED)
# onnxruntime
if(NOT DEFINED onnxruntime_DIR)
    set(onnxruntime_DIR "${LIB_ROOT}/onnxruntime/lib/cmake/onnxruntime")
endif()
find_package(onnxruntime REQUIRED)

add_compile_definitions(FMT_HEADER_ONLY)

add_executable(cnn_ncnn src/cnn_ncnn.cpp)
target_link_libraries(cnn_ncnn PRIVATE ncnn OpenMP::OpenMP_CXX)

add_executable(cnn_ort src/cnn_ort.cpp)
target_link_libraries(cnn_ort PRIVATE onnxruntime::onnxruntime)
