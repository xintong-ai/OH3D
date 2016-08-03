#include "GLSphere.h"
#include "vector_types.h"
#include "vector_functions.h"
//for linux
#define _USE_MATH_DEFINES
#include <cmath>
//using namespace std;
#include "helper_math.h"

GLSphere::GLSphere(float r, int n)
{
	//build the regular mesh in the constructor
	//using namespace std;
	float x, y, z;
	float n_rev = 1.0f / n;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float3 v0 = normalize(make_float3(0.5, i * n_rev - 0.5, j * n_rev - 0.5));
			float3 v1 = normalize(make_float3(0.5, (i + 1) * n_rev - 0.5, j * n_rev - 0.5));
			float3 v2 = normalize(make_float3(0.5, i * n_rev - 0.5, (j + 1) * n_rev - 0.5));
			float3 v3 = normalize(make_float3(0.5, (i + 1) * n_rev - 0.5, (j + 1) * n_rev - 0.5));
			grid.push_back(v0);
			grid.push_back(v1);
			grid.push_back(v3);
			grid.push_back(v2);
		}
	}
	//mirror by the plane
	for (int i = 0; i < 4 * n * n; i++) {
		grid.push_back(make_float3(-grid[i].x, grid[i].y, grid[i].z));
	}
	//rotate x, y, z
	for (int i = 0; i < 8 * n * n; i++) {
		grid.push_back(make_float3(grid[i].z, grid[i].x, grid[i].y));
	}
	for (int i = 0; i < 8 * n * n; i++) {
		grid.push_back(make_float3(grid[i].y, grid[i].z, grid[i].x));
	}
}

