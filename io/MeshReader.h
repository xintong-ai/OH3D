#ifndef MESH_READER_H
#define MESH_READER_H

#include <fstream>
#include <iostream>
#define _USE_MATH_DEFINES
#include "math.h"
#include "vector_types.h"

struct MeshReader
{
	int TotalConnectedTriangles;
	//int TotalConnectedQuads;
	int TotalConnectedPoints;
	int TotalFaces;

	float* Faces_Triangles = 0;
	float* Normals = 0;
	unsigned int* indices = 0;
	int numElements;

	void LoadPLY(const char* filename);

	void SphereMesh(float radius, unsigned int rings, unsigned int sectors);
	
	float3 center;
	void computeCenter(); //this center is the average position of all face centers, NOT average of vertices positions

	~MeshReader(){
		if (Faces_Triangles) delete[] Faces_Triangles;
		if (Normals) delete[] Normals;
		if (indices) delete[] indices;
	}
};



#endif