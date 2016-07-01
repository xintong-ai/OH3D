cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)

PROJECT (ParticleVis)

find_package(OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)

option(USE_OSVR "Enable features that require OSVR" OFF)
if(USE_OSVR)
	find_package(osvr REQUIRED)
endif()

option(USE_LEAP "Enable features that require LEAP MOTION" OFF)
if(USE_LEAP)
	find_package(Leap REQUIRED)
endif()

include_directories(
	${SHARED_LIB_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CUDA_TOOLKIT_INCLUDE} 
	${CUDA_SDK_ROOT_DIR}/common/inc 
	${Leap_INCLUDE_DIR}
	${PROJECT_BINARY_DIR}
	required)

set( Srcs 
	main.cpp 
	window.cpp 
	# glwidget.cpp 
	# Renderable.cpp 
	# trackball.cpp 
	# Rotation.cpp 
	# QuatVector.cpp
	# DataMgr.cpp 
	# Reader.cpp 
	# BoxRenderable.cpp 
	# GlyphRenderable.cpp 
	# LensRenderable.cpp
	# GridRenderable.cpp 
    # GLMatrixManager.cpp
    # ModelGridRenderable.cpp
	# GLSphere.cpp 
	)
set( Hdrs  
	window.h 
	# glwidget.h 
	# Renderable.h 
	# trackball.h 
	# Rotation.h 
	# QuatVector.h
	# DataMgr.h 
	# Reader.h 
	# BoxRenderable.h 
	# ShaderProgram.h 
	# GlyphRenderable.h 
	# LensRenderable.h
	# GridRenderable.h
	# ColorGradient.h 
	# GLMatrixManager.h
	# ModelGridRenderable.h
	# OPENGL_DRIVER.h
	# GLSphere.h
    )

	# LIST(APPEND Hdrs 
		# SphereRenderable.h 
		# ParticleReader.h 
	# )

	# LIST(APPEND Srcs 
		# ParticleReader.cpp 
		# SphereRenderable.cpp
	# )

# if(USE_OSVR)
	# set(Hdrs ${Hdrs} VRWidget.h	VRGlyphRenderable.h)
	# set(Srcs ${Srcs} VRWidget.cpp VRGlyphRenderable.cpp)
# endif()

# if(USE_LEAP)
	# set(Hdrs ${Hdrs} LeapListener.h LeapInteraction.h)
	# set(Srcs ${Srcs} LeapListener.cpp)
# endif()

configure_file(config.txt config.txt COPYONLY)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

add_executable(render ${Srcs} ${Hdrs})
#qt5_use_modules(render OpenGL Gui Widgets)
target_link_libraries(${PROJECT_NAME} 
	Qt5::Widgets  
	${OPENGL_LIBRARIES} 
	deform_lib
	)

if(USE_OSVR)
	target_link_libraries(render osvr::osvrClientKitCpp)
endif()

if(USE_LEAP)
	target_link_libraries(render ${Leap_IMPORT_LIB_DEBUG})
endif()
