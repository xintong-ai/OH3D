cmake_minimum_required(VERSION 2.6)

PROJECT (DeformVis)

find_package(Qt5Widgets REQUIRED)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

option(USE_SUPERQUADRIC "Enable features that require TEEM" OFF)

find_package(GLM REQUIRED)

if(USE_SUPERQUADRIC)
	find_package(Teem REQUIRED)
endif()

option(USE_LEAP OFF)

option(USE_OSVR "Enable features that require OSVR" OFF)

if(USE_OSVR)
	if (Boost_FOUND)
	else ()
	set ( BOOST_ROOT "/usr/local" CACHE PATH "the root directory containing Boost")
	endif ()

	find_package(osvr REQUIRED)
	include_directories(${osvr_DIR}/../../../include)
endif()

option(USE_CONTROLLER "Enable VIVE controller. !! NOT tested yet!!" OFF)

option(USE_TOUCHSCREEN "Enable touch screen interaction. !! NOT tested yet!!" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

set(DEFORM_DIR		"${CMAKE_SOURCE_DIR}/deform")
set(INTERACT_DIR	"${CMAKE_SOURCE_DIR}/interact")
set(IO_DIR			"${CMAKE_SOURCE_DIR}/io")
set(RENDER_DIR		"${CMAKE_SOURCE_DIR}/render")
set(DATAMODEL_DIR		"${CMAKE_SOURCE_DIR}/dataModel")
set(VPSEL_DIR		"${CMAKE_SOURCE_DIR}/vpsel")

include_directories(
	${DEFORM_DIR}
	${INTERACT_DIR}
	${VPSEL_DIR}
	${IO_DIR}
	${RENDER_DIR}
	${GLM_INCLUDE_DIR}
	${DATAMODEL_DIR}
	cuda_types/
	)

add_subdirectory(vpsel)

add_subdirectory(deform)

add_subdirectory(dataModel)

add_subdirectory(interact)

add_subdirectory(io)

add_subdirectory(render)

add_subdirectory(programs)

configure_file (
  "${PROJECT_SOURCE_DIR}/CMakeConfig.h.in"
  "${PROJECT_BINARY_DIR}/CMakeConfig.h"
  )