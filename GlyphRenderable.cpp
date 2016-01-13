#include "GlyphRenderable.h"
#include <Displace.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <glwidget.h>
#include <Lens.h>
#include <LensRenderable.h>

void GlyphRenderable::ComputeDisplace()
{
	displace->Compute(&matrix_mv.v[0].x, &matrix_pj.v[0].x, winWidth, winHeight, 
		((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses(), pos);
}

GlyphRenderable::GlyphRenderable(float4* _pos, int _num) 
{ 
	pos = _pos; 
	num = _num; 
	displace = new Displace();
	displace->LoadOrig(pos, num); 
}

void GlyphRenderable::RecomputeTarget()
{
	displace->RecomputeTarget();
}

//float3 GlyphRenderable::DataCenter()
//{
//	return (dataMin + dataMax) * 0.5;
//}


