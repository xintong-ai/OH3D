cmake_minimum_required(VERSION 2.6)

find_package(CUDA REQUIRED)


option(USE_SUPERQUADRIC "Enable features that require TEEM" OFF)
if(USE_SUPERQUADRIC)
	find_package(Teem REQUIRED)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

set(DEFORM_DIR		"${CMAKE_SOURCE_DIR}/deform_lib")
set(INTERACT_DIR	"${CMAKE_SOURCE_DIR}/interact")
set(IO_DIR			"${CMAKE_SOURCE_DIR}/io")
set(RENDER_DIR		"${CMAKE_SOURCE_DIR}/render")

include_directories(
	${DEFORM_DIR}
	${INTERACT_DIR}
	${IO_DIR}
	${RENDER_DIR}
	)

add_subdirectory(deform_lib)

add_subdirectory(interact)

add_subdirectory(io)

add_subdirectory(render)

add_subdirectory(programs)

configure_file (
  "${PROJECT_SOURCE_DIR}/CMakeConfig.h.in"
  "${PROJECT_BINARY_DIR}/CMakeConfig.h"
  )