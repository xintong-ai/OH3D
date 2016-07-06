cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

PROJECT(programs)

add_subdirectory(ParticleVis)

if(USE_SUPERQUADRIC)
	add_subdirectory(TensorVis)
endif()
# add_subdirectory(StreamVis)
