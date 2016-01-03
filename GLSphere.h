
#ifndef GLSPHERE_H
#define GLSPHERE_H
#include "vector_types.h"
#include <vector>

class GLSphere 
{
	std::vector<float3> grid;
public:
	explicit GLSphere(float r = 1.0f, int n = 16);
	int GetNumVerts(){ return grid.size(); }
	float* GetVerts(){
		float* ret = nullptr;
		if (grid.size() > 0)
			ret = (float*)&grid[0];
		return ret;
	}
};

#endif
