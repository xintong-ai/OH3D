cmake_minimum_required(VERSION 2.6)
project(interact)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)

if(TARGET interact)

message("interact is already defined")

else()

include_directories(${SHARED_LIB_INCLUDE_DIR})

set(HDRS	mouse/QuatVector.h mouse/Rotation.h mouse/trackball.h )
set(SRCS mouse/QuatVector.cpp mouse/Rotation.cpp mouse/trackball.cpp )

if(USE_DEFORM)
	set(HDRS ${HDRS} Lens.h)
	set(SRCS ${SRCS} Lens.cpp)
endif()

option(USE_LEAP "Enable Leap Motion motion camera" OFF)
if(USE_LEAP)
	set(HDRS ${HDRS} leap/LeapInteraction.h	leap/LeapListener.h)
	set(SRCS ${SRCS} leap/LeapListener.cpp)
	find_package(Qt5Widgets REQUIRED)
	#qt5_use_modules(render OpenGL Gui Widgets)
endif()


option(USE_CONTROLLER "Enable VIVE controller" OFF)
if(USE_CONTROLLER)
	find_package(Qt5Widgets REQUIRED)
	include_directories(${Boost_INCLUDE_DIR})
	set(HDRS ${HDRS} controller/Controller.h controller/QController.h)
	set(SRCS ${SRCS} controller/Controller.cpp controller/QController.cpp)
endif()

option(USE_TOUCHSCREEN "Enable touch screen interaction" OFF)
if(USE_TOUCHSCREEN)
endif()

add_library(interact STATIC ${HDRS} ${SRCS})
if(USE_LEAP)
	target_link_libraries(${PROJECT_NAME} ${Leap_IMPORT_LIB_DEBUG})
endif()

#if(USE_LEAP OR USE_CONTROLLER)
	target_link_libraries(${PROJECT_NAME} Qt5::Widgets)
#endif

endif()