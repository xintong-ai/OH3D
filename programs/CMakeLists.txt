cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

PROJECT(programs)
	add_subdirectory(ParticleVis)
	add_subdirectory(VolumeVis)
	
	if(USE_SUPERQUADRIC)
		add_subdirectory(TensorVis)
	endif()

	add_subdirectory(ImmersiveDeformVis)
	add_subdirectory(ImmersiveDeformPoly)
	add_subdirectory(ITKProcessing)
	add_subdirectory(PanaVis)
	
if(USE_LEAP)
	find_package(Qt5Widgets REQUIRED)
	#qt5_use_modules(render OpenGL Gui Widgets)
endif()
