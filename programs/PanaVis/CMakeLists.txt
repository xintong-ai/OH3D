cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
PROJECT (PanaVis)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)

find_package(OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)


include_directories(
	${SHARED_LIB_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CUDA_TOOLKIT_INCLUDE} 
	${CUDA_SDK_ROOT_DIR}/common/inc 
	${PROJECT_BINARY_DIR}
	${CMAKE_BINARY_DIR}
	required)
		
set( Srcs 
	main.cpp 
	window.cpp
	imageComputer.cpp
	)

set( Hdrs  
	window.h
	imageComputer.h
    )

configure_file(config.txt config.txt COPYONLY)

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
	dataModel
	deform
	${ITK_LIBRARIES}
)


if(USE_LEAP)
	include_directories(${Leap_ROOT_DIR}/include)
	target_link_libraries(${PROJECT_NAME} ${Leap_IMPORT_LIB_DEBUG})
endif()

if(USE_OSVR)
	target_link_libraries(${PROJECT_NAME} osvr::osvrClientKitCpp)
endif()
