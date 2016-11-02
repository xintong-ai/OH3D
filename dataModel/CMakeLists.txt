cmake_minimum_required(VERSION 2.6)
project(dataModel)
set(SHARED_LIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(SHARED_LIB_INCLUDE_DIR ${SHARED_LIB_INCLUDE_DIR} PARENT_SCOPE)

if(TARGET dataModel)

message("dataModel is already defined")

else()

include_directories(${SHARED_LIB_INCLUDE_DIR})

set(SRCS Volume.cpp Particle.cpp)

set(HDRS Volume.h Particle.h Processor.h)

add_library(dataModel STATIC ${HDRS} ${SRCS})

endif()