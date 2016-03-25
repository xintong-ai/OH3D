project(glyph_prog)

cmake_minimum_required(VERSION 2.6)

#Set(CMAKE_PREFIX_PATH "C:/Program Files (x86)/Windows Kits/8.0/Lib/win8/um/x64")

find_package(OpenGL REQUIRED)
find_package(Qt5Widgets)
find_package(CUDA REQUIRED)
find_package(Teem)
find_package(Leap)
find_package(osvr)

#find_package(OpenMP)
#if(OPENMP_FOUND)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} 
	${CUDA_TOOLKIT_INCLUDE} ${CUDA_SDK_ROOT_DIR}/common/inc 
	${Teem_INCLUDE_DIRS} 
	${Leap_INCLUDE_DIR}
	required)

set(SHARED_LIBRARIES
		${OPENGL_LIBRARIES}
)

link_directories(${Teem_LIBRARY_DIRS})

set( Srcs main.cpp GLSphere.cpp GLArrow.cpp
	window.cpp glwidget.cpp Renderable.cpp trackball.cpp Rotation.cpp QuatVector.cpp
	DataMgr.cpp Reader.cpp 
	BoxRenderable.cpp ParticleReader.cpp SphereRenderable.cpp
	Displace.cu Lens.cpp GlyphRenderable.cpp LensRenderable.cpp
	GridRenderable.cpp SQRenderable.cpp ArrowRenderable.cpp
	VolumeReader.cpp DTIVolumeReader.cpp VecReader.cpp
	VRWidget.cpp VRGlyphRenderable.cpp
	LeapListener.cpp 
    GLMatrixManager.cpp
    ModelGridRenderable.cpp
	PolyRenderable.cpp
	MeshReader.cpp
	ModelGrid.cu
	)
set( Hdrs  
GLSphere.h
	window.h glwidget.h Renderable.h trackball.h Rotation.h QuatVector.h
	DataMgr.h defines.h Reader.h GLArrow.h
	BoxRenderable.h ShaderProgram.h ParticleReader.h SphereRenderable.h
	Displace.h TransformFunc.h Lens.h GlyphRenderable.h LensRenderable.h
	GridRenderable.h
	ColorGradient.h SQRenderable.h ArrowRenderable.h 
	VolumeReader.h DTIVolumeReader.h VecReader.h
	VRWidget.h VRGlyphRenderable.h
	LeapListener.h LeapInteraction.h
	GLMatrixManager.h
	ModelGridRenderable.h
	ModelGrid.h
	GridMesh.h
	OPENGL_DRIVER.h
	PolyRenderable.h
	MeshReader.h
	deform/CUDA_PROJECTIVE_TET_MESH.h
    )
#set( Rcs resources/vechist.qrc)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

#set(omp_deps gomp)

#list( APPEND CUDA_NVCC_FLAGS  -Xcompiler /openmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP )
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
#qt5_add_resources(QtApp_RCC_SRCS ${Rcs})
cuda_add_executable(${PROJECT_NAME} ${Srcs} ${Hdrs} ${QtApp_RCC_SRCS})
qt5_use_modules(${PROJECT_NAME} OpenGL Gui)
target_link_libraries(${PROJECT_NAME} Qt5::Widgets  ${SHARED_LIBRARIES} ${Teem_LIBRARY_DIRS}/teem.lib
${Leap_IMPORT_LIB_DEBUG} osvr::osvrClientKitCpp)
