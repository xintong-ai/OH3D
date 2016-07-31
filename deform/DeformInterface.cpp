#include <DeformInterface.h>
#include <Displace.h>

DeformInterface::DeformInterface()
{
	displace = new Displace();
}

void DeformInterface::Compute(float* modelview, float* projection, int winW, int winH,
	std::vector<Lens*> lenses, float4* ret, float* glyphSizeScale, 
	float* glyphBright, bool isFreezingFeature, int snappedGlyphId, int snappedFeatureId)
{
	displace->Compute(modelview, projection, winW, winH,
		lenses, ret, glyphSizeScale, glyphBright, isFreezingFeature,
		snappedGlyphId, snappedFeatureId);
}
void DeformInterface::LoadOrig(float4* v, int num){
	displace->LoadOrig(v, num);
}
void DeformInterface::LoadFeature(char* f, int num){
	displace->LoadFeature(f, num);
}
void DeformInterface::RecomputeTarget(){
	displace->RecomputeTarget();
}

void DeformInterface::DisplacePoints(std::vector<float2>& pts, std::vector<Lens*> lenses, float* modelview, float* projection, int winW, int winH)
{
	displace->DisplacePoints(pts, lenses, modelview, projection, winW, winH);
}

float3 DeformInterface::findClosetGlyph(float3 aim, int &snappedGlyphId)
{
	return displace->findClosetGlyph(aim, snappedGlyphId);
}
