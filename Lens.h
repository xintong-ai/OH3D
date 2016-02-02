#ifndef LENS_H
#define LENS_H
#include <vector_types.h>
#include <helper_math.h>
#include <vector>
enum LENS_TYPE{
	TYPE_CIRCLE,
	TYPE_LINE,
};
struct Lens
{
	LENS_TYPE type;
	float3 c; //center
	int x, y; //screen location
	float focusRatio;
	float sideSize;
	float4 GetCenter();// { return make_float4(c.x, c.y, c.z, 1.0f); }
	void SetCenter(float3 _c){ c = _c; }
	void SetFocusRatio(float _v){ focusRatio = _v; }
	void SetSideSize(float _v){ sideSize = _v; }
	float GetClipDepth(float* mv, float* pj);
	Lens(int _x, int _y, float3 _c, float _focusRatio = 0.6, float _sideSize = 0.5) 
	{
		x = _x; y = _y; c = _c; focusRatio = _focusRatio; sideSize = _sideSize;
	}
	virtual bool PointInsideLens(int x, int y) = 0;
	virtual std::vector<float2> GetContour() = 0;
	virtual std::vector<float2> GetOuterContour() = 0;
	void ChangeClipDepth(int v, float* mv, float* pj);
	LENS_TYPE GetType(){ return type; }
};

struct CircleLens :public Lens
{
	float radius;
	CircleLens(int _x, int _y, int _r, float3 _c, float _focusRatio = 0.5, float _sideSize = 0.5) : Lens(_x, _y, _c, _focusRatio, _sideSize){ radius = _r; type = LENS_TYPE::TYPE_CIRCLE; };
	bool PointInsideLens(int _x, int _y) {
		float dis = length(make_float2(_x, _y) - make_float2(x, y));
		return dis < radius;
	}

	std::vector<float2> GetContourTemplate(int rr) {
		std::vector<float2> ret;
		const int num_segments = 32;
		for (int ii = 0; ii < num_segments; ii++)
		{
			float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);//get the current angle 

			float ax = rr * cosf(theta);//calculate the x component 
			float ay = rr * sinf(theta);//calculate the y component 

			ret.push_back(make_float2(x + ax, y + ay));//output vertex 
		}
		return ret;
	}

	std::vector<float2> GetContour() {
		return GetContourTemplate(radius);
	}

	std::vector<float2> GetOuterContour() {
		return GetContourTemplate(radius / focusRatio);
	}


};


struct LineLens :public Lens
{
	float lSemiMajorAxis, lSemiMinorAxis;
	float2 direction; //suppose normalized
	float ratio; //the ratio of lSemiMajorAxis and lSemiMinorAxis

	LineLens(int _x, int _y, int _r, float3 _c) : Lens(_x, _y, _c){ 
		lSemiMajorAxis = _r;
		ratio = 3.0f;
		lSemiMinorAxis = lSemiMajorAxis / ratio;

		direction = make_float2(1.0, 0.0);
		type = LENS_TYPE::TYPE_LINE;
	};

	bool PointInsideLens(int _x, int _y) {
		//sigmoid function: y=2*(1/(1+e^(-20*(x+1)))-0.5), x in [-1,0]
		//sigmoid function: y=2*(1/(1+e^(20*(x-1)))-0.5), x in [0,1]

		//dot product of (_x-x, _y-y) and direction

		float2 toPoint = make_float2(_x - x, _y - y);
		float disMajor = toPoint.x*direction.x + toPoint.y*direction.y;

		if (abs(disMajor) < lSemiMajorAxis) {
			float2 minorDirection = make_float2(-direction.y, direction.x);
			//dot product of (_x-x, _y-y) and minorDirection
			float disMinor = (_x - x)*minorDirection.x + (_y - y)*minorDirection.y;
			
			float disMajorRatio = disMajor / lSemiMajorAxis;
			float disSigmoid;
			if (disMajorRatio < 0){
				disSigmoid = 1 / (1 + exp(-40 * (disMajorRatio + 0.8))) ;
			}
			else {
				disSigmoid = 1 / (1 + exp(40 * (disMajorRatio - 0.8))) ;
			}

			if (abs(disMinor) < disSigmoid*lSemiMinorAxis)
				return true;
		}
		return false;
	}

	std::vector<float2> GetOuterContour() {
		std::vector<float2> ret;
		return ret;
	}

	std::vector<float2> GetContour() {
		//sigmoid function: y=2*(1/(1+e^(-20*(x+1)))-0.5), x in [-1,0]
		//sigmoid function: y=2*(1/(1+e^(20*(x-1)))-0.5), x in [0,1]
		float sigmoidCutOff = 0.4f; // assuming the sigmoid function value is constant when input is larger than sigmoidCutOff

		float2 minorDirection = make_float2(-direction.y, direction.x);

		std::vector<float2> ret;

		const int num_segments = 20;
		for (int ii = 0; ii < num_segments; ii++)
		{
			float tt = -1.0f + sigmoidCutOff*ii / num_segments;

			ret.push_back(make_float2(x, y) + tt*lSemiMajorAxis*direction
				+ (1 / (1 + exp(-40 * (tt + 0.8)))) *lSemiMinorAxis *minorDirection);//output vertex 
		}

		for (int ii = 0; ii < num_segments; ii++)
		{
			float tt = 1.0f - sigmoidCutOff + sigmoidCutOff*ii / num_segments;

			ret.push_back(make_float2(x, y) + tt*lSemiMajorAxis*direction
				+ (1 / (1 + exp(40 * (tt - 0.8)))) *lSemiMinorAxis *minorDirection);//output vertex 
		}

		
		for (int ii = 0; ii < num_segments; ii++)
		{
			float tt = 1.0f - sigmoidCutOff*ii / num_segments;

			ret.push_back(make_float2(x, y) + tt*lSemiMajorAxis*direction
				- (1 / (1 + exp(40 * (tt - 0.8)))) *lSemiMinorAxis *minorDirection);//output vertex 
		}

		for (int ii = 0; ii < num_segments; ii++)
		{
			float tt = -1.0f + sigmoidCutOff - sigmoidCutOff*ii / num_segments;

			ret.push_back(make_float2(x, y) + tt*lSemiMajorAxis*direction
				- (1 / (1 + exp(-40 * (tt + 0.8)))) *lSemiMinorAxis *minorDirection);//output vertex 
		}
		
		return ret;
	}
};
#endif