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
	unsigned int vertexcount = 0;
	unsigned int facecount = 0;
	
	float* vertexCoords = 0;
	float* vertexNorms = 0;
	unsigned int* indices = 0;

	bool readyForDeform = false;
	unsigned int vertexcountOri = 0;
	unsigned int facecountOri = 0;
	float* vertexCoordsOri = 0;
	float* vertexNormsOri = 0;
	unsigned int* indicesOri = 0;
	bool* faceValid = 0;

	~PolyMesh();

	float opacity = 1.0;
	void find_center_and_range();

	void GetPosRange(float3& posMin, float3& posMax);
	void setVertexCoordsOri();
	void reset();
	bool inRange(float3 v);

private:
	float cx, cy, cz;
	float min_x, max_x, min_y, max_y, min_z, max_z;

};
#endif