#pragma once
#include <vector>
#include <vector_types.h>
class Lens;
class Displace;
class DeformInterface
{
	Displace* displace;
public:
	DeformInterface();

	void Compute(float* modelview, float* projection, int winW, int winH,
		std::vector<Lens*> lenses, float4* ret, float* glyphSizeScale = 0, float* glyphBright = 0, bool isFreezingFeature = false, int snappedGlyphId = -1, int snappedFeatureId = -1);
	void LoadOrig(float4* v, int num);
	void LoadFeature(char* f, int num);
	void RecomputeTarget();// { recomputeTarget = true; }

	void DisplacePoints(std::vector<float2>& pts, std::vector<Lens*> lenses, float* modelview, float* projection, int winW, int winH);

	float3 findClosetGlyph(float3 aim, int &snappedGlyphId);


};