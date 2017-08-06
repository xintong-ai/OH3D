cmake_minimum_required(VERSION 2.6)
project(vpsel)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)

find_package(CUDA REQUIRED)

list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-DVERBOSE")
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

if(TARGET vpsel)

message("vpsel is already defined")

else()

include_directories(${SHARED_LIB_INCLUDE_DIR} 
	${CUDA_TOOLKIT_INCLUDE} 
	${CUDA_SDK_ROOT_DIR}/common/inc )


set(SRCS 	ViewpointEvaluator.cu
	)

set(HDRS		ViewpointEvaluator.h)

cuda_add_library(vpsel STATIC ${HDRS} ${SRCS})

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
#list(APPEND CUDA_NVCC_FLAGS "-g;-G")

endif()
