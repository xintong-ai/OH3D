#ifndef POLYMESH_H
#define POLYMESH_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

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

	float* vertexDeviateVals = 0; //scalar used for coloring the deformed part of the data. only used for deformation projects
	void setVertexDeviateVals();

	float* vertexColorVals = 0; //scalar used for coloring. may not be needed, depended on the polyRenderable class
	void setVertexColorVals(float v);


	bool readyForDeform = false;
	unsigned int vertexcountOri = 0;
	unsigned int facecountOri = 0;
	float* vertexCoordsOri = 0;
	float* vertexNormsOri = 0;
	unsigned int* indicesOri = 0;

	~PolyMesh();

	void createByCombiningPolyMeshes(std::vector<std::shared_ptr<PolyMesh>> polyMeshes);

	float opacity = 1.0;
	void find_center_and_range();

	void GetPosRange(float3& posMin, float3& posMax);
	void SetPosRange(float3 posMin, float3 posMax)
	{
		min_x = posMin.x, max_x = posMax.x, min_y = posMin.y, max_y = posMax.y, min_z = posMin.z, max_z = posMax.z;
	}

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
	

	static void dataParameters(std::string dataPath, float & disThr) //when subfolder is not needed
	{
		std::string subfolder_useless;
		dataParameters(dataPath, disThr, subfolder_useless);
	}

	static void dataParameters(std::string dataPath, float & disThr, std::string &subfolder)
	{
		if (std::string(dataPath).find("reduced-rbcs") != std::string::npos){
			disThr = 4.1;
			std::size_t found = dataPath.find_last_of("/\\");
			if (found == std::string::npos){
				subfolder = dataPath;//rarely
			}
			else{
				subfolder = dataPath.substr(0, found); //in the same folder
			}
		}
		else if (std::string(dataPath).find("rbcs") != std::string::npos){
			disThr = 4.1;
			subfolder = "bloodCell";
		}
		else if (std::string(dataPath).find("iso_t40_v3") != std::string::npos){
			disThr = 2;
		}
		else if (std::string(dataPath).find("moortgat") != std::string::npos){
			disThr = 2.1; //in a minimal case, this number should be no less than sqrt(3)/2
		}
		else if (std::string(dataPath).find("testDummy") != std::string::npos){
			disThr = 4;
		}	
		else{
			std::cout << "poly data name not recognized" << std::endl;
			exit(0);
		}
	};


	bool justChanged = false;
	std::shared_ptr<PolyMesh> newPoly;
	void createTestDummy();
private:
	float cx, cy, cz;
	float min_x, max_x, min_y, max_y, min_z, max_z;

};
#endif