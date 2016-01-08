#ifndef DISPLACE_H
#define DISPLACE_H
#include <thrust\device_vector.h>

class Displace
{
	thrust::device_vector<float3> posOrig;
public:
	Displace() {}
	void Compute(float* modelview, float* projection, float3* ret);
	void LoadOrig(float3* v, int num);
};

#endif
