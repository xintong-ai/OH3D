#ifndef LENS_H
#define LENS_H
#include <vector_types.h>

class Lens
{
	float3 c; //center
	int x, y; //screen location
public:
	float4 GetCenter() { make_float4(c.x, c.y, c.z, 1.0f); }
	Lens(int _x, int _y, float3 _c) { x = _x; y = _y; c = _c; }
};

class CircleLens :public Lens
{
	float radius;
public:
	CircleLens(int _x, int _y, int _r, float3 _c) : Lens(_x, _y, _c){ radius = _r; };
};

#endif