project(glyph_prog)

cmake_minimum_required(VERSION 2.6)

#Set(CMAKE_PREFIX_PATH "C:/Program Files (x86)/Windows Kits/8.0/Lib/win8/um/x64")

find_package(OpenGL REQUIRED)
find_package(Qt5Widgets)
find_package(CUDA REQUIRED)
#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} 
	${CUDA_TOOLKIT_INCLUDE} ${CUDA_SDK_ROOT_DIR}/common/inc required)

set(SHARED_LIBRARIES
		${OPENGL_LIBRARIES}
)

set( Srcs main.cpp GLSphere.cpp
	window.cpp glwidget.cpp Renderable.cpp Trackball.cpp Rotation.cpp QuatVector.cpp
	DataMgr.cpp Reader.cpp 
	BoxRenderable.cpp ParticleReader.cpp SphereRenderable.cpp
	Displace.cu Lens.cpp GlyphRenderable.cpp LensRenderable.cpp
	GridRenderable.cpp
	)
set( Hdrs 
GLSphere.h
	window.h glwidget.h Renderable.h Trackball.h Rotation.h QuatVector.h
	DataMgr.h defines.h Reader.h
	BoxRenderable.h ShaderProgram.h ParticleReader.h SphereRenderable.h
	Displace.h TransformFunc.h Lens.h GlyphRenderable.h LensRenderable.h
	GridRenderable.h
	ColorGradient.h
    )
#set( Rcs resources/vechist.qrc)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

#qt5_add_resources(QtApp_RCC_SRCS ${Rcs})
cuda_add_executable(${PROJECT_NAME} ${Srcs} ${Hdrs} ${QtApp_RCC_SRCS})
qt5_use_modules(${PROJECT_NAME} OpenGL Gui)
target_link_libraries(${PROJECT_NAME} Qt5::Widgets  ${SHARED_LIBRARIES} )
