cmake_minimum_required(VERSION 2.6)
project(io)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)

if(TARGET io)

message("io is already defined")

else()

include_directories(${SHARED_LIB_INCLUDE_DIR})

target_link_libraries(
	dataModel
	)

set(SRCS DataMgr.cpp MeshReader.cpp ParticleReader.cpp
	VecReader.cpp TeemVolumeReader.cpp DTIVolumeReader.cpp 
	SolutionParticleReader.cpp BinaryParticleReader.cpp
	RawVolumeReader.cpp
	BinaryTuplesReader.cpp)

set(HDRS DataMgr.h MeshReader.h ParticleReader.h
	VecReader.h TeemVolumeReader.h DTIVolumeReader.h 
	SolutionParticleReader.h BinaryParticleReader.h
	RawVolumeReader.h
	BinaryTuplesReader.h)

add_library(io STATIC ${HDRS} ${SRCS})

endif()