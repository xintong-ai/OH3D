#ifndef DISPLACE_H
#define DISPLACE_H
#include <thrust\device_vector.h>
class Lens;
class Displace
{
	thrust::device_vector < float4 > posOrig;
	thrust::device_vector<float2> d_vec_posScreenTarget;
	//thrust::device_vector < float4 > posScreenTarget;
	bool recomputeTarget = false;
public:
	Displace() {}
	void Compute(float* modelview, float* projection, 
		int winW, int winH, std::vector<Lens*> lenses, float4* ret);
	void LoadOrig(float4* v, int num);
	void RecomputeTarget(){ recomputeTarget = true; }
};

#endif
