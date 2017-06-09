#ifndef MESH_READER_H
#define MESH_READER_H

#include <fstream>
#include <iostream>
#include <memory>
#define _USE_MATH_DEFINES
#include "math.h"
#include "vector_types.h"
class PolyMesh;
/*
this is a reader written by Xin.
as a temporary solution, it currently only supports ply files in ascii, triangle faces with normal input.
refer to our VTK readers, or official ply readers if this class cannot satisfy the requirement
*/
class PlyMeshReader
{
public:
	int TotalConnectedTriangles;
	int TotalConnectedPoints;

	float* Faces_Triangles = 0;
	float* Normals = 0;
	unsigned int* indices = 0;

	void LoadPLY(const char* filename, std::shared_ptr<PolyMesh> polyMesh);
		
	float3 center;
	void computeCenter(); //this center is the average position of all face centers, NOT average of vertices positions

	~PlyMeshReader(){
	}
};



#endif