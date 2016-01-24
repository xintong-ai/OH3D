#include "GlyphRenderable.h"
#include <Displace.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <glwidget.h>
#include <Lens.h>
#include <LensRenderable.h>

//GlyphRenderable::GlyphRenderable(){}

void GlyphRenderable::ComputeDisplace()
{
	int2 winSize = actor->GetWindowSize();
	displace->Compute(&matrix_mv.v[0].x, &matrix_pj.v[0].x, winSize.x, winSize.y,
		((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses(), pos, &glyphSizeScale[0]);
}


GlyphRenderable::GlyphRenderable(float4* _pos, int _num)
{ 
	pos = _pos; 
	num = _num; 
	displace = std::make_shared<Displace>();
	displace->LoadOrig(pos, num); 
	glyphSizeScale.assign(num, 1.0f);
}

void GlyphRenderable::RecomputeTarget()
{
	displace->RecomputeTarget();
}

void GlyphRenderable::DisplacePoints(std::vector<float2>& pts)
{
	displace->DisplacePoints(pts,
		((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses());
}

//float3 GlyphRenderable::DataCenter()
//{
//	return (dataMin + dataMax) * 0.5;
//}

void GlyphRenderable::SlotFocusSizeChanged(int v)
{ 
	displace->SetFocusRatio( (10 - v) * 0.1 * 0.8 + 0.2); 
	displace->RecomputeTarget();
	actor->UpdateGL();
}

void GlyphRenderable::SlotSideSizeChanged(int v)// { displace - (10 - v) * 0.1; }
{
	displace->SetSideSize(v * 0.1); 
	displace->RecomputeTarget();
	actor->UpdateGL();
}

void GlyphRenderable::SlotGlyphSizeAdjustChanged(int v)// { displace - (10 - v) * 0.1; }
{
	glyphSizeAdjust = v * 0.1;
	actor->UpdateGL();
}
