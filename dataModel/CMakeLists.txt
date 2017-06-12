cmake_minimum_required(VERSION 2.8)
project(dataModel)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)

find_package(Qt5Widgets REQUIRED)


if(TARGET dataModel)
message("dataModel is already defined")
else()

include_directories(${SHARED_LIB_INCLUDE_DIR})

set(SRCS Volume.cpp Particle.cpp GLMatrixManager.cpp
	LabelVolumeProcessor.cpp
	AnimationByMatrixProcessor.cpp
	)
set(HDRS Volume.h Particle.h Processor.h myDefine.h GLMatrixManager.h ColorGradient.h ScreenMarker.h
	LabelVolumeProcessor.h
	AnimationByMatrixProcessor.h
	MatrixManager.h
	myDefineRayCasting.h
	PolyMesh.h
)
add_library(${PROJECT_NAME}  STATIC ${HDRS} ${SRCS})

target_link_libraries(${PROJECT_NAME} 
	Qt5::Widgets  
	)

endif()