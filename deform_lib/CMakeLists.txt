cmake_minimum_required(VERSION 2.6)
project(deform_lib)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)

if(TARGET deform_lib)

message("deform_lib is already defined")

else()

include_directories(${SHARED_LIB_INCLUDE_DIR} ${CUDA_TOOLKIT_INCLUDE} 
	${CUDA_SDK_ROOT_DIR}/common/inc )

set(LIB_SRCS ./src/Displace.cu ./src/ModelGrid.cu ./src/Lens.cpp)

cuda_add_library(deform_lib STATIC ${LIB_SRCS})

endif()