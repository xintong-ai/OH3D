cmake_minimum_required(VERSION 2.6)
project(interact)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)

if(TARGET interact)

message("interact is already defined")

else()

include_directories(${SHARED_LIB_INCLUDE_DIR}
	${CMAKE_BINARY_DIR}
)


set(HDRS	mouse/QuatVector.h mouse/Rotation.h mouse/trackball.h 
mouse/Interactor.h mouse/ImmersiveInteractor.h mouse/RegularInteractor.h 
mouse/MatrixInteractor.h 
mouse/LensInteractor.h mouse/GlyphInteractor.h
)

set(SRCS mouse/QuatVector.cpp mouse/Rotation.cpp mouse/trackball.cpp 
mouse/ImmersiveInteractor.cpp 
mouse/RegularInteractor.cpp
mouse/LensInteractor.cpp mouse/GlyphInteractor.cpp
)

add_definitions(-DEXPORT_QT)



if(USE_LEAP)
	find_package(Leap)
	include_directories(${Leap_ROOT_DIR}/include)
	set( HDRS ${HDRS} leap/LeapInteraction.h leap/LeapListener.h 
		leap/LeapInteractor.h leap/LensLeapInteractor.h)
	set( SRCS ${SRCS} leap/LeapListener.cpp leap/LensLeapInteractor.cpp)
endif()

if(USE_CONTROLLER)
#	find_package(Qt5Widgets REQUIRED)
	include_directories(${Boost_INCLUDE_DIR})
	set(HDRS ${HDRS} controller/Controller.h controller/QController.h)
	set(SRCS ${SRCS} controller/Controller.cpp controller/QController.cpp)
endif()

if(USE_TOUCHSCREEN)
	set( HDRS ${HDRS} touch/TouchInteractor.h touch/LensTouchInteractor.h)
	set( SRCS ${SRCS} touch/LensTouchInteractor.cpp)
endif()

#qt5_use_modules(${PROJECT_NAME} Qt5::Widgets)
SET(CMAKE_AUTOMOC ON)

add_library(interact STATIC ${HDRS} ${SRCS})


#if(USE_CONTROLLER)
	target_link_libraries(${PROJECT_NAME} Qt5::Widgets)
#endif

endif()