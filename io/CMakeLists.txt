cmake_minimum_required(VERSION 2.6)
project(io)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)

if(TARGET io)

message("io is already defined")

else()

include_directories(${SHARED_LIB_INCLUDE_DIR} ${CUDA_TOOLKIT_INCLUDE} 
	${CUDA_SDK_ROOT_DIR}/common/inc )


set(SRCS DataMgr.cpp MeshReader.cpp ParticleReader.cpp Reader.cpp 
	VecReader.cpp VolumeReader.cpp DTIVolumeReader.cpp 
	FlightReader.cpp UniverseParticleReader.cpp SolutionParticleReader.cpp)

set(HDRS DataMgr.h MeshReader.h ParticleReader.h Reader.h 
	VecReader.h VolumeReader.h DTIVolumeReader.h 
	FlightReader.h UniverseParticleReader.h SolutionParticleReader.h)

add_library(io STATIC ${HDRS} ${SRCS})

endif()