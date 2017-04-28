cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

PROJECT(programs)

add_subdirectory(ParticleVis)
add_subdirectory(VolumeVis)
add_subdirectory(ImmersiveVolumeVis)

if(USE_SUPERQUADRIC)
	add_subdirectory(TensorVis)
endif()
