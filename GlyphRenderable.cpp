#include "GlyphRenderable.h"
#include <Displace.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

void GlyphRenderable::ComputeDisplace()
{
	displace->Compute(&matrix_mv.v[0].x, &matrix_pj.v[0].x, winWidth, winHeight, pos);
}

GlyphRenderable::GlyphRenderable(float4* _pos, int _num) 
{ 
	pos = _pos; 
	num = _num; 
	displace = new Displace();
	displace->LoadOrig(pos, num); 
}

void GlyphRenderable::AddCircleLens()
{
	displace->AddSphereLens(winWidth * 0.5, winHeight * 0.5, winHeight * 0.1, DataCenter());
}


float3 GlyphRenderable::DataCenter()
{
	return (dataMin + dataMax) * 0.5;
}
