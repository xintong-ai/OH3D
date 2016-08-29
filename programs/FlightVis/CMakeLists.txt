cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)

PROJECT (FlightVis)

find_package(OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)



# option(USE_LEAP "Enable features that require LEAP MOTION" OFF)
if(USE_LEAP)
	include_directories(${LEAP_INCLUDE_DIR})
endif()

include_directories(
	${SHARED_LIB_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${PROJECT_BINARY_DIR}
	${CMAKE_BINARY_DIR}
	required)

set( Srcs 
	main.cpp 
	window.cpp 
	)
set( Hdrs  
	window.h 
    )

if(WIN32)
    configure_file(config.txt config.txt COPYONLY)
elseif(APPLE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/Debug/config.txt COPYONLY)
    elseif(CMAKE_BUILD_TYPE TREQUAL "Release")
        configure_file(config.txt ${CMAKE_CURRENT_BINARY_DIR}/Release/config.txt COPYONLY)
    endif()
endif(WIN32)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

add_executable(${PROJECT_NAME} ${Srcs} ${Hdrs} config.txt)
qt5_use_modules(${PROJECT_NAME} OpenGL Gui Widgets)
target_link_libraries(${PROJECT_NAME} 
	Qt5::Widgets  
	${OPENGL_LIBRARIES} 
	io
	interact
	render
	)

if(USE_OSVR)
	target_link_libraries(${PROJECT_NAME} osvr::osvrClientKitCpp)
endif()

if(USE_LEAP)
	target_link_libraries(${PROJECT_NAME} ${Leap_IMPORT_LIB_DEBUG})
endif()
