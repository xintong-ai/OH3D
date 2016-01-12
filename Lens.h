#ifndef LENS_H
#define LENS_H
#include <vector_types.h>
#include <helper_math.h>
#include <vector>
struct Lens
{
	float3 c; //center
	int x, y; //screen location
	float4 GetCenter();// { return make_float4(c.x, c.y, c.z, 1.0f); }
	void SetCenter(float3 _c){ c = _c; }
	float GetClipDepth(float* mv, float* pj);
	Lens(int _x, int _y, float3 _c) { x = _x; y = _y; c = _c; }
	virtual bool PointInsideLens(int x, int y) = 0;
	virtual std::vector<float2> GetContour() = 0;
	void ChangeClipDepth(int v, float* mv, float* pj);
};

struct CircleLens :public Lens
{
	float radius;
	CircleLens(int _x, int _y, int _r, float3 _c) : Lens(_x, _y, _c){ radius = _r; };
	bool PointInsideLens(int _x, int _y) {
		float dis = length(make_float2(_x, _y) - make_float2(x, y));
		return dis < radius;
	}

	std::vector<float2> GetContour() {
		std::vector<float2> ret;
		const int num_segments = 32;
		for (int ii = 0; ii < num_segments; ii++)
		{
			float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);//get the current angle 

			float ax = radius * cosf(theta);//calculate the x component 
			float ay = radius * sinf(theta);//calculate the y component 

			ret.push_back(make_float2(x + ax, y + ay));//output vertex 
		}
		return ret;
	}


};

#endif