cmake_minimum_required(VERSION 2.6)
project(interact)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)

if(TARGET interact)

message("interact is already defined")

else()

include_directories(${SHARED_LIB_INCLUDE_DIR})

set(HDRS	mouse/QuatVector.h mouse/Rotation.h mouse/trackball.h)
set(SRCS mouse/QuatVector.cpp mouse/Rotation.cpp mouse/trackball.cpp)

option(USE_LEAP "Enable Leap Motion motion camera" OFF)
if(USE_LEAP)
	set(HDRS ${HDRS} leap/LeapInteraction.h	leap/LeapListener.h)
	set(SRCS ${SRCS} leap/LeapListener.cpp)
	find_package(Qt5Widgets REQUIRED)
	#qt5_use_modules(render OpenGL Gui Widgets)
endif()

option(USE_TOUCHSCREEN "Enable touch screen interaction" OFF)
if(USE_TOUCHSCREEN)
endif()

add_library(interact STATIC ${HDRS} ${SRCS})
if(USE_LEAP)
	target_link_libraries(${PROJECT_NAME} ${Leap_IMPORT_LIB_DEBUG} Qt5::Widgets)
endif()

endif()