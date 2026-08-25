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

add_executable(cnn_struct src/cnn_struct.cpp)
target_include_directories(cnn_struct PRIVATE ${OPENBLAS_INCLUDE_DIR})
target_link_directories(cnn_struct PRIVATE ${OPENBLAS_LIB_DIR})
target_link_libraries(cnn_struct PRIVATE -lopenblas OpenMP::OpenMP_CXX -lm)
target_compile_options(cnn_struct PRIVATE -funsafe-math-optimizations -mcpu=native)

add_executable(cnn_const src/cnn_const.cpp)
target_include_directories(cnn_const PRIVATE ${OPENBLAS_INCLUDE_DIR})
target_link_directories(cnn_const PRIVATE ${OPENBLAS_LIB_DIR})
target_link_libraries(cnn_const PRIVATE -lopenblas OpenMP::OpenMP_CXX -lm)
target_compile_options(cnn_const PRIVATE -funsafe-math-optimizations -mcpu=native)

add_executable(cnn_neon src/cnn_neon.cpp)
target_link_libraries(cnn_neon PRIVATE OpenMP::OpenMP_CXX -lm)
target_compile_options(cnn_neon PRIVATE -funsafe-math-optimizations -mcpu=native)

add_executable(cnn_fuse src/cnn_fuse.cpp)
target_link_libraries(cnn_fuse PRIVATE OpenMP::OpenMP_CXX -lm)
target_compile_options(cnn_fuse PRIVATE -funsafe-math-optimizations -mcpu=native)
