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
class Particle;

class PolyMesh
{
public:
	unsigned int vertexcount = 0;
	unsigned int facecount = 0;
	
	float* vertexCoords = 0;
	float* vertexNorms = 0;
	unsigned int* indices = 0;

	float* vertexColorVals = 0; //scalar used for coloring. may not be needed for many case. how to use it is also case by case
	void setVertexColorVals(float v = 0);

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

	//particle is optional. needed when the poly mesh is constituted by several connected components.
	//the count of the particles stored in the object is the same with the count of the connected components.
	//the center of the connected components are stored in pos, and their properties are stored in valTuple
	//when this object is used, the coordinate stored in vertexCoords[] is the relative position of each vertex from the component center
	std::shared_ptr<Particle> particle = 0; 
	void setAssisParticle(const char* fname);

	void doShift(float3 shift);
	
	static void dataParameters(std::string dataPath, int3 & dims, float3 &spacing, float & disThr, float3 &shift, std::string &subfolder)
	{
		if (std::string(dataPath).find("iso_t40_v3") != std::string::npos){
			dims = make_int3(68, 68, 68);
			spacing = make_float3(1, 1, 1);
			disThr = 2;
			shift = make_float3(ceil(disThr) + 1, ceil(disThr) + 1, ceil(disThr) + 1); //+1 for more margin
			subfolder = "FPM";
		}
		else{
			std::cout << "volume data name not recognized" << std::endl;
			exit(0);
		}
	};

private:
	float cx, cy, cz;
	float min_x, max_x, min_y, max_y, min_z, max_z;

};
#endif