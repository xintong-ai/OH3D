cmake_minimum_required(VERSION 2.6)

option(USE_SUPERQUADRIC "Enable features that require TEEM" OFF)
if(USE_SUPERQUADRIC)
	find_package(Teem REQUIRED)
endif()

option(USE_OSVR "Enable features that require OSVR" OFF)
if(USE_OSVR)
	find_package(osvr REQUIRED)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

set(DEFORM_DIR		"${CMAKE_SOURCE_DIR}/deform")
set(INTERACT_DIR	"${CMAKE_SOURCE_DIR}/interact")
set(IO_DIR			"${CMAKE_SOURCE_DIR}/io")
set(RENDER_DIR		"${CMAKE_SOURCE_DIR}/render")
set(DATAMODEL_DIR		"${CMAKE_SOURCE_DIR}/dataModel")

include_directories(
	${DEFORM_DIR}
	${INTERACT_DIR}
	${IO_DIR}
	${RENDER_DIR}
	${DATAMODEL_DIR}
	)

option(USE_DEFORM "Enable deformation" ON)
if(USE_DEFORM)
	add_subdirectory(deform)
endif()


add_subdirectory(interact)

add_subdirectory(io)

add_subdirectory(dataModel)

add_subdirectory(render)

add_subdirectory(programs)

configure_file (
  "${PROJECT_SOURCE_DIR}/CMakeConfig.h.in"
  "${PROJECT_BINARY_DIR}/CMakeConfig.h"
  )