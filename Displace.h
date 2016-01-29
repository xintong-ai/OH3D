#ifndef DISPLACE_H
#define DISPLACE_H
#include <thrust/device_vector.h>
class Lens;
class Displace
{
	thrust::device_vector < float4 > posOrig;
	thrust::device_vector<float2> d_vec_posScreenTarget;
	thrust::device_vector<float> d_vec_glyphSizeTarget;
	//thrust::device_vector<float> d_vec_Dist2LensBtm;
	//thrust::device_vector < float4 > posScreenTarget;
	bool recomputeTarget = false;
	//float focusRatio = 0.6;
	//float sideSize = 0.5;
public:
	Displace();
	void Compute(float* modelview, float* projection, int winW, int winH,
		std::vector<Lens*> lenses, float4* ret, float* glyphSizeScale = 0);
	void LoadOrig(float4* v, int num);
	void RecomputeTarget(){ recomputeTarget = true; }
	void DisplacePoints(std::vector<float2>& pts, std::vector<Lens*> lenses);
	//void SetFocusRatio(float v) { focusRatio = v; }
	//void SetSideSize(float v) { sideSize = v; }
};

#endif
