#ifndef DISPLACE_H
#define DISPLACE_H
#include <thrust/device_vector.h>
class Lens;
class Displace
{
	thrust::device_vector < float4 > posOrig;
	thrust::device_vector<float4> d_vec_posTarget;
	thrust::device_vector<float> d_vec_glyphSizeTarget;
	thrust::device_vector<float> d_vec_glyphBrightTarget;
	bool recomputeTarget = true;

	thrust::device_vector<float> d_vec_disToAim; //used for snapping

public:
	Displace();
	void Compute(float* modelview, float* projection, int winW, int winH,
		std::vector<Lens*> lenses, float4* ret, float* glyphSizeScale = 0, float* glyphBright = 0);
	void LoadOrig(float4* v, int num);
	void RecomputeTarget(){ recomputeTarget = true; }

	void DisplacePoints(std::vector<float2>& pts, std::vector<Lens*> lenses, float* modelview, float* projection, int winW, int winH);


	float3 findClosetGlyph(float3 aim, int &snappedGlyphId);



};

#endif
