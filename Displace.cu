#include <Displace.h>

void Displace::LoadOrig(float3* v, int num)
{
	posOrig.assign(v, v + num);// , posOrig.begin());
}

void Displace::Compute(float* modelview, float* projection, float3* ret)
{
	thrust::copy(posOrig.begin(), posOrig.end(), ret);
}
