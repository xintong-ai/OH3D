#include <Particle.h>
#include <iostream>

#include <fstream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>


Particle::Particle(std::vector<float4> _pos, std::vector<float> _val)
{
	pos = _pos;
	posOrig = _pos;
	val = _val;
	numParticles = pos.size();

	posMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	posMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	float v = 0;
	for (int i = 0; i < pos.size(); i++) {
		v = pos[i].x;
		if (v > posMax.x)
			posMax.x = v;
		if (v < posMin.x)
			posMin.x = v;

		v = pos[i].y;
		if (v > posMax.y)
			posMax.y = v;
		if (v < posMin.y)
			posMin.y = v;

		v = pos[i].z;
		if (v > posMax.z)
			posMax.z = v;
		if (v < posMin.z)
			posMin.z = v;
	}

	valMax = -FLT_MAX;
	valMin = FLT_MAX;
	for (int i = 0; i < val.size(); i++) {
		v = val[i];
		if (v > valMax)
			valMax = v;
		if (v < valMin)
			valMin = v;
	}
}
