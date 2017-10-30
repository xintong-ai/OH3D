cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
PROJECT (AssistProg)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)

find_package(OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)

# for itk
INCLUDE(${CMAKE_SOURCE_DIR}/programs/AssistProg/IJMacros.txt)
SET(Required_Packages
  ITK
)
FOREACH(Package ${Required_Packages})
  LOADPACKAGE(${Package})
ENDFOREACH(Package)


include_directories(
	${SHARED_LIB_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CUDA_TOOLKIT_INCLUDE} 
	${CUDA_SDK_ROOT_DIR}/common/inc 
	${PROJECT_BINARY_DIR}
	${CMAKE_BINARY_DIR}
	required)
		
set( SRCS 
	main.cpp 
	)

configure_file(config.txt config.txt COPYONLY)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

add_executable(${PROJECT_NAME} ${SRCS} ${HDRS} config.txt)

qt5_use_modules(${PROJECT_NAME} OpenGL Gui Widgets)
target_link_libraries(${PROJECT_NAME} 
	Qt5::Widgets  
	${OPENGL_LIBRARIES} 
	io
	dataModel
	deform #now this one is required since volume.h contains VolumeCUDA, which needs cuda components
	${ITK_LIBRARIES}
)

if(USE_OSVR)
	target_link_libraries(${PROJECT_NAME} osvr::osvrClientKitCpp)
endif()


if(USE_VTK)
	set(HDRS ${HDRS} PlyVTKReader.h VTPReader.h)
	set(SRCS ${SRCS} PlyVTKReader.cpp VTPReader.cpp)
	include(${VTK_USE_FILE}) 
endif()

if(USE_VTK)
	target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
endif()

find_package(EIGEN REQUIRED)
include_directories(${EIGEN_INCLUDE_DIR} required)