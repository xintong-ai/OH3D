cmake_minimum_required(VERSION 2.6)

PROJECT (DeformVis)

find_package(Qt5Widgets REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

option(BUILD_TUTORIAL "Build a tutorial project" ON)

option(BUILD_TEST "Build ongoing projects that are not well mainteined" OFF)


option(USE_TEEM "Enable features that require Teem" OFF)
if(USE_TEEM)
	find_package(Teem REQUIRED)
endif()

option(USE_GLM "Enable features that require glm" ON)
if(USE_GLM)
	find_package(GLM REQUIRED)
endif()


option(USE_LEAP OFF)
if(USE_LEAP)
	find_package(Leap)
endif()

option(USE_OSVR "Enable features that require OSVR" OFF)
if(USE_OSVR)
	if (Boost_FOUND)
	else ()
	set ( BOOST_ROOT "/usr/local" CACHE PATH "the root directory containing Boost")
	endif ()

	find_package(osvr REQUIRED)
	include_directories(${osvr_DIR}/../../../include)
endif()

option(USE_TOUCHSCREEN "Enable touch screen interaction." OFF)



if(BUILD_TEST)
	option(USE_VTK "Enable features that require VTK" OFF)
	if(USE_VTK)
		find_package(VTK REQUIRED)
	endif()

	option(USE_ITK "Enable features that require ITK" OFF)
	if(USE_ITK)
		find_package(ITK REQUIRED)
	endif()
endif ()

option(USE_CONTROLLER "Enable VIVE controller. !! NOT tested yet!!" OFF)

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

if(USE_ITK)
	add_subdirectory(vpsel)
	#essentially vpsel does not need ITK. but all executables using vpsel will use ITK
endif()

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