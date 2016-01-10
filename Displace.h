#ifndef DISPLACE_H
#define DISPLACE_H
#include <thrust\device_vector.h>

class Displace
{
	thrust::device_vector<float4> posOrig;
public:
	Displace() {}
	void Compute(float* modelview, float* projection, int winW, int winH, float4* ret);
	void LoadOrig(float4* v, int num);
};

#endif
