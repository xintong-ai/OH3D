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
//written by Xin?
class PolyMesh
{
public:

	unsigned int vertexcount = 0;
	unsigned int facecount = 0;
	
	float* vertexCoords = 0;
	float* vertexCoordsOri = 0;

	float* vertexNorms = 0;
	unsigned int* indices = 0;

	~PolyMesh(){
		if (vertexCoords) delete[]vertexCoords;
		if (vertexCoordsOri) delete[]vertexCoordsOri;
		if (vertexNorms) delete[]vertexNorms;
		if (indices) delete[]indices;
	}

	float opacity = 1.0;

	void find_center_and_range()
	{
		float x, y, z;
		min_x = 9999, max_x = -9999, min_y = 9999, max_y = -9999;
		min_z = 9999, max_z = -9999;

		x = y = z = 0;
		for (int i = 0; i < vertexcount; i++) {
			x += vertexCoords[3 * i];
			y += vertexCoords[3 * i + 1];
			z += vertexCoords[3 * i + 2];
			if (min_x > vertexCoords[3 * i]) min_x = vertexCoords[3 * i];
			if (max_x < vertexCoords[3 * i]) max_x = vertexCoords[3 * i];
			if (min_y > vertexCoords[3 * i + 1]) min_y = vertexCoords[3 * i + 1];
			if (max_y < vertexCoords[3 * i + 1]) max_y = vertexCoords[3 * i + 1];
			if (min_z > vertexCoords[3 * i + 2]) min_z = vertexCoords[3 * i + 2];
			if (max_z < vertexCoords[3 * i + 2]) max_z = vertexCoords[3 * i + 2];
		}
		cx = x / (float)vertexcount;
		cy = y / (float)vertexcount;
		cz = z / (float)vertexcount;

	}

	void GetPosRange(float3& posMin, float3& posMax)
	{
		posMin = make_float3(min_x, min_y, min_z);
		posMax = make_float3(max_x, max_y, max_z);
	}

	void setVertexCoordsOri()
	{
		if (vertexcount > 0){
			if (vertexCoordsOri) delete[]vertexCoordsOri;
			vertexCoordsOri = (float*)malloc(sizeof(float)* 3 * vertexcount);
			memcpy(vertexCoordsOri, vertexCoords, sizeof(float)* 3 * vertexcount);
		}
	}


	void reset()
	{
		if (vertexcount > 0){
			memcpy(vertexCoords, vertexCoordsOri, sizeof(float)* 3 * vertexcount);
		}
	};

private:
	float cx, cy, cz;
	float min_x, max_x, min_y, max_y, min_z, max_z;

};
#endif