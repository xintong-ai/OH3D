cmake_minimum_required(VERSION 2.6)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

add_subdirectory(deform_lib)

add_subdirectory(render)

add_subdirectory(interact)

add_subdirectory(io)

configure_file (
  "${PROJECT_SOURCE_DIR}/CMakeConfig.h.in"
  "${PROJECT_BINARY_DIR}/CMakeConfig.h"
  )