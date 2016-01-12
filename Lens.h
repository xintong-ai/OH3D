#ifndef LENS_H
#define LENS_H
#include <vector_types.h>

struct Lens
{
	float3 c; //center
	int x, y; //screen location
	float4 GetCenter();// { return make_float4(c.x, c.y, c.z, 1.0f); }
	float GetClipDepth(float* mv, float* pj);
	Lens(int _x, int _y, float3 _c) { x = _x; y = _y; c = _c; }
};

struct CircleLens :public Lens
{
	float radius;
	CircleLens(int _x, int _y, int _r, float3 _c) : Lens(_x, _y, _c){ radius = _r; };
};

#endif