cmake_minimum_required(VERSION 2.6)
project(io)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)

if(TARGET io)

message("io is already defined")

else()

include_directories(${SHARED_LIB_INCLUDE_DIR})


set(SRCS DataMgr.cpp PlyMeshReader.cpp  ParticleReader.cpp
	VecReader.cpp NrrdVolumeReader.cpp DTIVolumeReader.cpp 
	SolutionParticleReader.cpp BinaryParticleReader.cpp
	RawVolumeReader.cpp
	BinaryTuplesReader.cpp
	)

set(HDRS DataMgr.h PlyMeshReader.h  ParticleReader.h
	VecReader.h NrrdVolumeReader.h DTIVolumeReader.h 
	SolutionParticleReader.h BinaryParticleReader.h
	RawVolumeReader.h
	BinaryTuplesReader.h
	)

if(USE_VTK)
	set(HDRS ${HDRS} PlyVTKReader.h )
	set(SRCS ${SRCS} PlyVTKReader.cpp)
	include(${VTK_USE_FILE}) 
endif()

add_library(${PROJECT_NAME}  STATIC ${HDRS} ${SRCS})

if(USE_VTK)
	target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
endif()

endif()