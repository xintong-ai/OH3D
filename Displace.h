#ifndef DISPLACE_H
#define DISPLACE_H
#include <thrust\device_vector.h>
class Lens;
class Displace
{
	thrust::device_vector < float4 > posOrig;
	//thrust::device_vector < float4 > posScreenTarget;
	bool doRefresh = false;
	std::vector<Lens*> lenses;
public:
	Displace() {}
	void Compute(float* modelview, float* projection, int winW, int winH, float4* ret);
	void LoadOrig(float4* v, int num);
	void AddSphereLens(int x, int y, int radius, float3 center);
};

#endif
