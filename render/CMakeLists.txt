cmake_minimum_required(VERSION 2.6)
project(render)
find_package(OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)

#if(USE_OSVR)
#	find_package(osvr)
#endif


include_directories(
	${SHARED_LIB_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CUDA_TOOLKIT_INCLUDE} 
	${CUDA_SDK_ROOT_DIR}/common/inc 
	# ${Leap_INCLUDE_DIR}
	${CMAKE_BINARY_DIR}
	${Boost_INCLUDE_DIR}
	${osvr_DIR}/../../../include
	required)

if(USE_SUPERQUADRIC)
	include_directories(
		${Teem_INCLUDE_DIRS} 
	)
	link_directories(${Teem_LIBRARY_DIRS})
endif()


set( Srcs 
	glwidget.cpp 
	Renderable.cpp 
	BoxRenderable.cpp 
	GlyphRenderable.cpp 
	LensRenderable.cpp
	GridRenderable.cpp 
	GLArrow.cpp
	ArrowRenderable.cpp
    GLMatrixManager.cpp
    ModelGridRenderable.cpp
	GLSphere.cpp 
	SphereRenderable.cpp
	PolyRenderable.cpp
	)
set( Hdrs  
	glwidget.h 
	Renderable.h 
	BoxRenderable.h 
	ShaderProgram.h 
	GlyphRenderable.h 
	LensRenderable.h
	GridRenderable.h
	ColorGradient.h 
	GLArrow.h
	ArrowRenderable.h 
	GLMatrixManager.h
	ModelGridRenderable.h
	OPENGL_DRIVER.h
	GLSphere.h
	SphereRenderable.h 
    PolyRenderable.h
	)

if(USE_OSVR)
	set(Hdrs ${Hdrs} VRWidget.h	VRGlyphRenderable.h)
	set(Srcs ${Srcs} VRWidget.cpp VRGlyphRenderable.cpp)
endif()

if(USE_SUPERQUADRIC)
	set(Hdrs ${Hdrs} SQRenderable.h)
	set(Srcs ${Srcs} SQRenderable.cpp)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

add_library(${PROJECT_NAME} STATIC ${Srcs} ${Hdrs})
#qt5_use_modules(render OpenGL Gui Widgets)
target_link_libraries(${PROJECT_NAME} 
	Qt5::Widgets  
	${OPENGL_LIBRARIES} 
	deform
	interact
	)

if(USE_SUPERQUADRIC)
target_link_libraries(${PROJECT_NAME} 
	${Teem_LIBRARY_DIRS}/teem.lib)
endif()

# if(USE_OSVR)
	# target_link_libraries(${PROJECT_NAME} osvr::osvrClientKitCpp)
# endif()

# if(USE_LEAP)
	# target_link_libraries(${PROJECT_NAME} ${Leap_IMPORT_LIB_DEBUG})
# endif()
