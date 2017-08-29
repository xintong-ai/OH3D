cmake_minimum_required(VERSION 2.6)
project(render)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)


find_package(OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)

if(TARGET render)
message("render is already defined")
else()

find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-DVERBOSE")
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

include_directories(
	${SHARED_LIB_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_BINARY_DIR}
	${Boost_INCLUDE_DIR}
	 ${CUDA_TOOLKIT_INCLUDE} 
	required)

set( Srcs 
	glwidget.cpp 
	Renderable.cpp 
	BoxRenderable.cpp 
	GlyphRenderable.cpp 
	GLArrow.cpp
	ArrowRenderable.cpp
	GLSphere.cpp 
	SphereRenderable.cpp
	PolyRenderable.cpp
	GridRenderable.cpp
	VolumeRenderableImmerCUDA.cpp
	VolumeRenderableCUDA.cpp
	VolumeRenderableCUDAKernel.cu
	LensRenderable.cpp DeformGLWidget.cpp 
		MeshRenderable.cpp
		GLWidgetQtDrawing.cpp
		MatrixMgrRenderable.cpp
	#InfoGuideRenderable.cpp
		DeformFrameRenderable.cpp
		TraceRenderable.cpp
		SliceRenderable.cpp
	)
set( Hdrs  
	glwidget.h 
	Renderable.h 
	BoxRenderable.h 
	ShaderProgram.h 
	GlyphRenderable.h 
	GridRenderable.h
	GLArrow.h
	ArrowRenderable.h 
	GLSphere.h
	SphereRenderable.h 
    PolyRenderable.h
	SliceRenderable.h

	VolumeRenderableImmerCUDA.h
	VolumeRenderableCUDA.h
	VolumeRenderableCUDAKernel.h
	LensRenderable.h DeformGLWidget.h 	
		MeshRenderable.h
		GLWidgetQtDrawing.h
		MatrixMgrRenderable.h
	#InfoGuideRenderable.h
		DeformFrameRenderable.h
		TraceRenderable.h
	)

if(USE_TEEM)
	set(Hdrs ${Hdrs} SQRenderable.h)
	set(Srcs ${Srcs} SQRenderable.cpp)
	include_directories(${Teem_INCLUDE_DIRS})

	IF (WIN32)
		link_directories(${Teem_LIBRARY_DIRS}/Release)
	ELSE()
		link_directories(${Teem_LIBRARY_DIRS})
	ENDIF()
endif()

if(USE_OSVR)
	set(Hdrs ${Hdrs} VRWidget.h	VRGlyphRenderable.h VRVolumeRenderableCUDA.h)
	set(Srcs ${Srcs} VRWidget.cpp VRGlyphRenderable.cpp VRVolumeRenderableCUDA.cpp)
endif()


set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(CUDA_NVCC_FLAGS_DEBUG "-g -G")

cuda_add_library(${PROJECT_NAME} STATIC ${Srcs} ${Hdrs})

qt5_use_modules(${PROJECT_NAME} OpenGL Gui Widgets)
target_link_libraries(${PROJECT_NAME} 
	Qt5::Widgets  
	${OPENGL_LIBRARIES} 
	interact
	dataModel
	deform
	)

if(USE_TEEM)
	IF (WIN32)
		target_link_libraries(${PROJECT_NAME} 
			${Teem_LIBRARY_DIRS}/Release/teem.lib) #since the shared lib of teem is used, it is safe for Debug mode too
	ELSE()
		target_link_libraries(${PROJECT_NAME} 
			${Teem_LIBRARY_DIRS}/teem.lib)
	ENDIF()
endif()

if(USE_OSVR)
	target_link_libraries(${PROJECT_NAME} osvr::osvrClientKitCpp)
endif()

endif()