#
# Find GLM
#
# Try to find GLM : OpenGL Mathematics.
# This module defines 
# - GLM_INCLUDE_DIRS
# - GLM_FOUND
#
# The following variables can be set as arguments for the module.
# - GLM_ROOT_DIR : Root library directory of GLM 
#
# References:
# - https://github.com/Groovounet/glm/blob/master/util/FindGLM.cmake
# - https://bitbucket.org/alfonse/gltut/src/28636298c1c0/glm-0.9.0.7/FindGLM.cmake
#

# Additional modules
include(FindPackageHandleStandardArgs)

if (WIN32)
	# Find include files
	find_path(
		EIGEN_INCLUDE_DIR
		NAMES signature_of_eigen3_matrix_library
		PATHS
		$ENV{PROGRAMFILES}/include
		${EIGEN_ROOT_DIR}/include
		DOC "The directory where signature_of_eigen3_matrix_library resides")
else()
	# Find include files
	find_path(
		EIGEN_INCLUDE_DIR
		NAMES signature_of_eigen3_matrix_library
		PATHS
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		${EIGEN_ROOT_DIR}/include
		DOC "The directory where signature_of_eigen3_matrix_library resides")
endif()

# Handle REQUIRD argument, define *_FOUND variable
find_package_handle_standard_args(EIGEN DEFAULT_MSG EIGEN_INCLUDE_DIR)

# Define EIGEN_INCLUDE_DIRS
if (EIGEN_FOUND)
	set(EIGEN_INCLUDE_DIRS ${EIGEN_INCLUDE_DIR})
endif()

# Hide some variables
mark_as_advanced(EIGEN_INCLUDE_DIR)