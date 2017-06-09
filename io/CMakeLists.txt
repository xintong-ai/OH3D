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
	${VTK_LIBRARIES}
	)

set(SRCS DataMgr.cpp PlyMeshReader.cpp PlyVTKReader.cpp ParticleReader.cpp
	VecReader.cpp NrrdVolumeReader.cpp DTIVolumeReader.cpp 
	SolutionParticleReader.cpp BinaryParticleReader.cpp
	RawVolumeReader.cpp
	BinaryTuplesReader.cpp
	)

set(HDRS DataMgr.h PlyMeshReader.h  PlyVTKReader.h ParticleReader.h
	VecReader.h NrrdVolumeReader.h DTIVolumeReader.h 
	SolutionParticleReader.h BinaryParticleReader.h
	RawVolumeReader.h
	BinaryTuplesReader.h
	)

set(VTK_DIR "C:/Users/lc/Library/VTK-7.1.1/")
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})  


add_library(io STATIC ${HDRS} ${SRCS})

endif()