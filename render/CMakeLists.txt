cmake_minimum_required(VERSION 2.6)

find_package(OpenGL REQUIRED)
find_package(Qt5Widgets)
find_package(Teem)
find_package(Leap)
find_package(osvr)

include_directories(
	${SHARED_LIB_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CUDA_TOOLKIT_INCLUDE} 
	${CUDA_SDK_ROOT_DIR}/common/inc 
	${Teem_INCLUDE_DIRS} 
	${Leap_INCLUDE_DIR}
	required)

link_directories(${Teem_LIBRARY_DIRS})

set( Srcs 
	main.cpp 
	GLSphere.cpp 
	GLArrow.cpp
	window.cpp 
	glwidget.cpp 
	Renderable.cpp 
	trackball.cpp 
	Rotation.cpp 
	QuatVector.cpp
	DataMgr.cpp 
	Reader.cpp 
	BoxRenderable.cpp 
	ParticleReader.cpp 
	SphereRenderable.cpp
	GlyphRenderable.cpp 
	LensRenderable.cpp
	GridRenderable.cpp 
	SQRenderable.cpp 
	ArrowRenderable.cpp
	VolumeReader.cpp 
	DTIVolumeReader.cpp 
	VecReader.cpp
	VRWidget.cpp 
	VRGlyphRenderable.cpp
	LeapListener.cpp 
    GLMatrixManager.cpp
    ModelGridRenderable.cpp
	PolyRenderable.cpp
	MeshReader.cpp
	
	)
set( Hdrs  
	GLSphere.h
	window.h 
	glwidget.h 
	Renderable.h 
	trackball.h 
	Rotation.h 
	QuatVector.h
	DataMgr.h 
	Reader.h 
	GLArrow.h
	BoxRenderable.h 
	ShaderProgram.h 
	ParticleReader.h 
	SphereRenderable.h
	GlyphRenderable.h 
	LensRenderable.h
	GridRenderable.h
	ColorGradient.h 
	SQRenderable.h 
	ArrowRenderable.h 
	VolumeReader.h 
	DTIVolumeReader.h 
	VecReader.h
	VRWidget.h 
	VRGlyphRenderable.h
	LeapListener.h 
	LeapInteraction.h
	GLMatrixManager.h
	ModelGridRenderable.h
	OPENGL_DRIVER.h
	PolyRenderable.h
	MeshReader.h
    )

configure_file(config.txt config.txt COPYONLY)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

add_executable(render ${Srcs} ${Hdrs})
#qt5_use_modules(render OpenGL Gui Widgets)
target_link_libraries(render 
	Qt5::Widgets  
	${OPENGL_LIBRARIES} 
	${Teem_LIBRARY_DIRS}/teem.lib
	${Leap_IMPORT_LIB_DEBUG} 
	osvr::osvrClientKitCpp 
	deform_lib)
