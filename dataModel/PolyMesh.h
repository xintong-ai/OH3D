#ifndef POLYMESH_H
#define POLYMESH_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <math.h>
#include <memory>
//using namespace std;

#include <cuda_runtime.h>
#include <helper_cuda.h>

//currently only support triangles

class PolyMesh
{
public:

	unsigned int vertexcount;
	unsigned int facecount;
	int vertexnormals = 0;
	int facenormals = 0;
	
	float* vertexCoords = 0;
	float* vertexNorms = 0;
	unsigned int* indices = 0;

	~PolyMesh(){
		if (vertexCoords) delete[]vertexCoords;
		if (vertexNorms) delete[]vertexNorms;
		if (indices) delete[]indices;
	}

	void read(const char* fname);

private:
	float cx, cy, cz;
	float x_min, x_max, y_min, y_max, z_min, z_max;

	/*
	void find_center(float& cx, float& cy, float& cz,
		float& minx, float& maxx, float&miny,
		float &maxy, float &minz, float & maxz)
	{
		float x, y, z;
		float min_x = 9999, max_x = -9999, min_y = 9999, max_y = -9999;
		float min_z = 9999, max_z = -9999;

		x = y = z = 0;
		for (int i = 0; i < vertexcount; i++) {
			x += vertices[i]->x;
			y += vertices[i]->y;
			z += vertices[i]->z;
			if (min_x >vertices[i]->x) min_x = vertices[i]->x;
			if (max_x <vertices[i]->x) max_x = vertices[i]->x;
			if (min_y >vertices[i]->y) min_y = vertices[i]->y;
			if (max_y <vertices[i]->y) max_y = vertices[i]->y;
			if (min_z >vertices[i]->z) min_z = vertices[i]->z;
			if (max_z <vertices[i]->z) max_z = vertices[i]->z;
		}
		cx = x / (float)vertexcount;
		cy = y / (float)vertexcount;
		cz = z / (float)vertexcount;
		minx = min_x; maxx = max_x;
		miny = min_y; maxy = max_y;
		minz = min_z; maxz = max_z;
	}
	*/

};
#endif