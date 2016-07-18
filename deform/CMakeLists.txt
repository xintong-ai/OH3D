cmake_minimum_required(VERSION 2.6)
project(deform)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)

find_package(CUDA REQUIRED)

#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-DVERBOSE")
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

if(TARGET deform)

message("deform is already defined")

else()

include_directories(${SHARED_LIB_INCLUDE_DIR} ${CUDA_TOOLKIT_INCLUDE} 
	${CUDA_SDK_ROOT_DIR}/common/inc )


set(SRCS Displace.cu ModelGrid.cu Lens.cpp)

set(HDRS	defines.h
			Displace.h
			GridMesh.h
			LineSplitGridMesh.h
			Lens.h
			ModelGrid.h
			TransformFunc.h)

cuda_add_library(deform STATIC ${HDRS} ${SRCS})

endif()
