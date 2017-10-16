cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

PROJECT(programs)
add_subdirectory(ParticleVis)
add_subdirectory(VolumeVis)

if(BUILD_TUTORIAL)
	add_subdirectory(TutorialVis)
endif()

if(USE_TEEM)
	add_subdirectory(TensorVis)
endif()

if(BUILD_TEST)
	if(USE_ITK)
		#add_subdirectory(PanaVis)
		add_subdirectory(ImmersiveDeformVis)
		if(USE_VTK)
			add_subdirectory(ImmersiveDeformPoly)
			#add_subdirectory(ImmersiveDeformParticle)#temporarily to speed up for testing...
			add_subdirectory(ImmersiveDeformParticleTV)
			add_subdirectory(AssistProg)
		endif()
	endif()
endif()