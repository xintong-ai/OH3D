cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

PROJECT(programs)



	add_subdirectory(ParticleVis)
	#add_subdirectory(UniverseVis)
	
	if(USE_SUPERQUADRIC)
		add_subdirectory(TensorVis)
	endif()
	add_subdirectory(VolumeVis)
	add_subdirectory(ImmersiveVolumeVis)

if(USE_LEAP)
	find_package(Qt5Widgets REQUIRED)
	#qt5_use_modules(render OpenGL Gui Widgets)
endif()

#add_subdirectory(FlightVis)


# add_subdirectory(StreamVis)
