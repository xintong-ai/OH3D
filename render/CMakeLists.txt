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
	${CUDA_TOOLKIT_INCLUDE} 
	${CUDA_SDK_ROOT_DIR}/common/inc 
)

include_directories(
	${SHARED_LIB_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_BINARY_DIR}
	${Boost_INCLUDE_DIR}
	 ${CUDA_TOOLKIT_INCLUDE} 
	#${CUDA_SDK_ROOT_DIR}/common/inc
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
	GLArrow.cpp
	ArrowRenderable.cpp
	GLSphere.cpp 
	SphereRenderable.cpp
	PolyRenderable.cpp
	CosmoRenderable.cpp
	GridRenderable.cpp
	VolumeRenderableImmerCUDA.cpp
	VolumeRenderableCUDA.cpp
	VolumeRenderableCUDAKernel.cu
	LensRenderable.cpp DeformGLWidget.cpp 
		MeshRenderable.cpp
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
	CosmoRenderable.h
	VolumeRenderableImmerCUDA.h
	VolumeRenderableCUDA.h
	VolumeRenderableCUDAKernel.h
	LensRenderable.h DeformGLWidget.h 	
		MeshRenderable.h
		)


if(USE_OSVR)
	set(Hdrs ${Hdrs} VRWidget.h	VRGlyphRenderable.h VRVolumeRenderableCUDA.h)
	set(Srcs ${Srcs} VRWidget.cpp VRGlyphRenderable.cpp VRVolumeRenderableCUDA.cpp)
endif()


if(USE_SUPERQUADRIC)
	set(Hdrs ${Hdrs} SQRenderable.h)
	set(Srcs ${Srcs} SQRenderable.cpp)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

#set(CUDA_NVCC_FLAGS_DEBUG "-g -G")

cuda_add_library(${PROJECT_NAME} STATIC ${Srcs} ${Hdrs})

qt5_use_modules(${PROJECT_NAME} OpenGL Gui Widgets)
target_link_libraries(${PROJECT_NAME} 
	Qt5::Widgets  
	${OPENGL_LIBRARIES} 
	interact
	dataModel
	)


target_link_libraries(${PROJECT_NAME} 
	deform
	)


if(USE_SUPERQUADRIC)
target_link_libraries(${PROJECT_NAME} 
	${Teem_LIBRARY_DIRS}/teem.lib)
endif()

if(USE_OSVR)
	target_link_libraries(${PROJECT_NAME} osvr::osvrClientKitCpp)
endif()

endif()