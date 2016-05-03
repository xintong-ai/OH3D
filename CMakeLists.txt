project(glyph_prog)

cmake_minimum_required(VERSION 2.6)

#Set(CMAKE_PREFIX_PATH "C:/Program Files (x86)/Windows Kits/8.0/Lib/win8/um/x64")


find_package(CUDA REQUIRED)


#find_package(OpenMP)
#if(OPENMP_FOUND)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

add_subdirectory(deform_lib)

add_subdirectory(render)

#set( Rcs resources/vechist.qrc)



#set(omp_deps gomp)

#list( APPEND CUDA_NVCC_FLAGS  -Xcompiler /openmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP )
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
#qt5_add_resources(QtApp_RCC_SRCS ${Rcs})
