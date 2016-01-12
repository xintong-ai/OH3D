#include "Lens.h"
#include "TransformFunc.h"
#include <vector_functions.h>

float Lens::GetClipDepth(float* mv, float* pj)
{
	return Object2Clip(GetCenter(), mv, pj).z;
}

float4 Lens::GetCenter() { return make_float4(c.x, c.y, c.z, 1.0f); }
