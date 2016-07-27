cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

PROJECT(programs)


if(USE_DEFORM)
	add_subdirectory(ParticleVis)
	add_subdirectory(ParticleVis2)
	add_subdirectory(UniverseVis)
	if(USE_SUPERQUADRIC)
		add_subdirectory(TensorVis)
	endif()
endif()

add_subdirectory(FlightVis)


# add_subdirectory(StreamVis)
