
#ifndef GLARROW_H
#define GLARROW_H
#include "vector_types.h"
#include <vector>
#include "helper_math.h"

class GLArrow
{
public:

	std::vector<float4> grids;
	std::vector<float4> colors;
	std::vector<float3> normals;
	std::vector<unsigned int> indices;

	//paramteres
	float range = 0.5; //[-range,range]
	float width = 0.25; //[-width, width] for rod. tip width is decided by ratioTipRod 
	float ratioTipRod = 3;
	float rodRatio = 0.5;
	float tipRatio = 1 - rodRatio;
	float3 orientation = make_float3(0, 0, 1);
	const int nDivision = 16;

public:
	explicit GLArrow();

	int GetNumVerts(){ return grids.size(); }
	float* GetVerts(){
		float* ret = nullptr;
		if (grids.size() > 0)
			ret = (float*)&grids[0];
		return ret;
	}

	float* GetColors(){
		float* ret = nullptr;
		if (colors.size() > 0)
			ret = (float*)&colors[0];
		return ret;
	}

	float* GetNormals(){
		float* ret = nullptr;
		if (normals.size() > 0)
			ret = (float*)&normals[0];
		return ret;
	}

	unsigned int* GetIndices(){
		unsigned int* ret = nullptr;
		if (indices.size() > 0)
			ret = (unsigned int*)&indices[0];
		return ret;
	}

	int GetNumIndices(){ return indices.size(); }

};

#endif
