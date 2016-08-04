cmake_minimum_required(VERSION 2.6)
project(render)
find_package(OpenGL REQUIRED)
find_package(Qt5Widgets REQUIRED)



if(USE_VOLUME_RENDERING)
	find_package(CUDA REQUIRED)
	#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")
	list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-DVERBOSE")
	#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
	include_directories(
		${CUDA_TOOLKIT_INCLUDE} 
		${CUDA_SDK_ROOT_DIR}/common/inc 
	)
endif()



include_directories(
	${SHARED_LIB_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_BINARY_DIR}
	${Boost_INCLUDE_DIR}
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
    GLMatrixManager.cpp
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
	ColorGradient.h 
	GLArrow.h
	ArrowRenderable.h 
	GLMatrixManager.h
	#OPENGL_DRIVER.h
	GLSphere.h
	SphereRenderable.h 
    PolyRenderable.h
	)

if(USE_DEFORM)
	set(Hdrs ${Hdrs} DeformGlyphRenderable.h LensRenderable.h DeformGLWidget.h 	
		ModelGridRenderable.h GridRenderable.h)
	set(Srcs ${Srcs} DeformGlyphRenderable.cpp LensRenderable.cpp DeformGLWidget.cpp 
		ModelGridRenderable.cpp GridRenderable.cpp )
	include_directories(
		 ${CUDA_TOOLKIT_INCLUDE} 
	#${CUDA_SDK_ROOT_DIR}/common/inc
	)
endif()

if(USE_OSVR)
	set(Hdrs ${Hdrs} VRWidget.h	VRGlyphRenderable.h VRVolumeRenderableCUDA.h)
	set(Srcs ${Srcs} VRWidget.cpp VRGlyphRenderable.cpp VRVolumeRenderableCUDA.cpp)
endif()

if(USE_VOLUME_RENDERING)
	set(Hdrs ${Hdrs} VolumeRenderableCUDA.h
	VolumeRenderableCUDAKernel.h)
	set(Srcs ${Srcs} VolumeRenderableCUDA.cpp
	VolumeRenderableCUDAKernel.cu)
endif()

if(USE_SUPERQUADRIC)
	set(Hdrs ${Hdrs} SQRenderable.h)
	set(Srcs ${Srcs} SQRenderable.cpp)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

#set(CUDA_NVCC_FLAGS_DEBUG "-g -G")

if(USE_VOLUME_RENDERING)
	cuda_add_library(${PROJECT_NAME} STATIC ${Srcs} ${Hdrs})
else()
	add_library(${PROJECT_NAME} STATIC ${Srcs} ${Hdrs})
endif()

#qt5_use_modules(render OpenGL Gui Widgets)
target_link_libraries(${PROJECT_NAME} 
	Qt5::Widgets  
	${OPENGL_LIBRARIES} 
	interact
	dataModel
	)

if(USE_DEFORM)
target_link_libraries(${PROJECT_NAME} 
	deform
	)
endif()

if(USE_SUPERQUADRIC)
target_link_libraries(${PROJECT_NAME} 
	${Teem_LIBRARY_DIRS}/teem.lib)
endif()

if(USE_OSVR)
	target_link_libraries(${PROJECT_NAME} osvr::osvrClientKitCpp)
endif()
