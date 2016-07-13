#ifndef LENS_H
#define LENS_H
#include <vector_types.h>
#include <vector_functions.h>
#include <cmath>
using namespace std;
#include <helper_math.h>
#include <vector>

#include <iostream>
//using namespace std;

enum LENS_TYPE{
	TYPE_CIRCLE,
	TYPE_LINE,
	TYPE_LINEB, 
	TYPE_CURVEB,
};

struct LineBLensInfo
{
	float lSemiMajorAxis, lSemiMinorAxis;
	float2 direction; //suppose normalized
	float focusRatio;
};

struct CurveBLensInfo
{
	int numBezierPoints;
	float2 BezierPoints[25];

	int numPosPoints;
	float2 subCtrlPointsPos[100];
	float2 posOffsetCtrlPoints[100];
	int numNegPoints;
	float2 subCtrlPointsNeg[100];
	float2 negOffsetCtrlPoints[100];

	float width;
	float focusRatio;//ratio of focus region and transition region
};


struct Lens
{
	Lens(float3 _c, float _focusRatio = 0.6)//, float _sideSize = 0.5) //int _x, int _y, 
	{
		c = _c; focusRatio = _focusRatio; //sideSize = _sideSize;//x = _x; y = _y; 
	}

	const int eps_pixel = 32;
	LENS_TYPE type;
	float3 c; //center
	//int x, y; //screen location
	float focusRatio;
	//float sideSize;
	float4 GetCenter();// { return make_float4(c.x, c.y, c.z, 1.0f); }
	void SetCenter(float3 _c){ c = _c; }
	void SetFocusRatio(float _v){ focusRatio = _v; }
	//void SetSideSize(float _v){ sideSize = _v; }
	float GetClipDepth(float* mv, float* pj);
	float2 GetCenterScreenPos(float* mv, float* pj, int winW, int winH);
	void UpdateCenterByScreenPos(int sx, int sy, float* mv, float* pj, int winW, int winH);//update c by new screen position (sx,sy)
	float3 Compute3DPosByScreenPos(int sx, int sy, float* mv, float* pj, int winW, int winH);	

	virtual bool PointInsideLens(int _x, int _y, float* mv, float* pj, int winW, int winH) = 0;
	virtual bool PointInsideFullLens(int _x, int _y, float* mv, float* pj, int winW, int winH) { return false; }
	virtual bool PointOnInnerBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) { return false; }
	virtual bool PointOnOuterBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) { return false; }
	virtual bool PointInsideObjectLens(int x, int y, float* mv, float* pj, int winW, int winH) { return false; }

	virtual bool PointOnObjectInnerBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) { return false; }
	virtual bool PointOnObjectOuterBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) { return false; }
	//virtual bool PointOnCriticalPos(int _x, int _y, float* mv, float* pj, int winW, int winH) { return false; }
	virtual void ChangeLensTwoFingers(int2 p1, int2 p2, float* mv, float* pj, int winW, int winH){}
	virtual void ChangeLensSize(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH) {}
	virtual void ChangefocusRatio(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH) {}
	virtual void ChangeObjectLensSize(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH) {}	
	virtual void ChangeObjectFocusRatio(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH) {}
	//virtual void ChangeDirection(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH) {}
	virtual std::vector<float2> GetContour(float* mv, float* pj, int winW, int winH) = 0;
	virtual std::vector<float2> GetOuterContour(float* mv, float* pj, int winW, int winH) = 0;
	virtual std::vector<float2> GetCtrlPointsForRendering(float* mv, float* pj, int winW, int winH) = 0; //cannot directly use the ctrlPoints array, since need to haddle constructing process
	void ChangeClipDepth(int v, float* mv, float* pj);
	void SetClipDepth(float d, float* mv, float* pj);
	LENS_TYPE GetType(){ return type; }

	bool isConstructing;


	bool PointOnLensCenter(int _x, int _y, float* mv, float* pj, int winW, int winH) {
		float2 center = GetCenterScreenPos(mv, pj, winW, winH);
		float dis = length(make_float2(_x, _y) - center);// make_float2(x, y));
		return dis < eps_pixel;
	}
};

struct CircleLens :public Lens
{
	float radius;
	float objectRadius;

	CircleLens(int _r, float3 _c, float _focusRatio = 0.5) : Lens(_c, _focusRatio)
	{ 
		radius = _r;
		objectRadius = 1.3;
		type = LENS_TYPE::TYPE_CIRCLE;
		isConstructing = false;
	};
	

	bool PointInsideLens(int _x, int _y, float* mv, float* pj, int winW, int winH) {
		float2 center = GetCenterScreenPos(mv, pj, winW, winH);
		float dis = length(make_float2(_x, _y) - center);// make_float2(x, y));
		return dis < radius;
	}

	bool PointInsideFullLens(int _x, int _y, float* mv, float* pj, int winW, int winH)
	{
		float2 center = GetCenterScreenPos(mv, pj, winW, winH);
		float dis = length(make_float2(_x, _y) - center);// make_float2(x, y));
		return dis < (radius / focusRatio);
	}

	
	bool PointOnInnerBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) override
	{
		float2 center = GetCenterScreenPos(mv, pj, winW, winH);
		float dis = length(make_float2(_x, _y) - center);// make_float2(x, y));
		return std::abs(dis - radius) < eps_pixel;
	}

	bool PointOnOuterBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) override
	{ 
		float2 center = GetCenterScreenPos(mv, pj, winW, winH);
		float dis = length(make_float2(_x, _y) - center);// make_float2(x, y));
		return std::abs(dis - radius / focusRatio) < eps_pixel;
	}
	
	bool PointInsideObjectLens(int x, int y, float* mv, float* pj, int winW, int winH) override;
	bool PointOnObjectInnerBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) override;
	bool PointOnObjectOuterBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) override;

	void ChangeLensTwoFingers(int2 p1, int2 p2, float* mv, float* pj, int winW, int winH) override
	{
		float dis = length(make_float2(p1) - make_float2(p2));
		radius = dis * 0.5;
		float2 center = (make_float2(p1) + make_float2(p2)) * 0.5;
		UpdateCenterByScreenPos(center.x, center.y, mv, pj, winW, winH);
	}

	
	void ChangeLensSize(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH) override
	{
		float2 center = GetCenterScreenPos(mv, pj, winW, winH);
		float dis = length(make_float2(_x, _y) - center);// make_float2(x, y));
		radius = dis;
	}

	void ChangefocusRatio(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH) override
	{
		float2 center = GetCenterScreenPos(mv, pj, winW, winH);
		float dis = length(make_float2(_x, _y) - center);// make_float2(x, y));
		if (dis > radius + eps_pixel + 1)
			focusRatio = radius / dis; 
	}
	void ChangeObjectLensSize(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH) override;
	void ChangeObjectFocusRatio(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)override;

	std::vector<float2> GetContourTemplate(int rr, float* mv, float* pj, int winW, int winH) {
		std::vector<float2> ret;
		const int num_segments = 32;
		for (int ii = 0; ii < num_segments; ii++)
		{
			float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);//get the current angle 

			float ax = rr * cosf(theta);//calculate the x component 
			float ay = rr * sinf(theta);//calculate the y component 
			float2 center = GetCenterScreenPos(mv, pj, winW, winH);
			ret.push_back(make_float2(center.x + ax, center.y + ay));//output vertex 
		}
		return ret;
	}

	std::vector<float2> GetContour(float* mv, float* pj, int winW, int winH) {
		return GetContourTemplate(radius, mv, pj, winW, winH);
	}

	std::vector<float2> GetOuterContour(float* mv, float* pj, int winW, int winH) {
		return GetContourTemplate(radius / focusRatio, mv, pj, winW, winH);
	}

	std::vector<float2> GetCtrlPointsForRendering(float* mv, float* pj, int winW, int winH){
		std::vector<float2> res(0);
		return res;
	}

	std::vector<std::vector<float3>> Get3DContour(float3 eyeWorld, bool isScreenDeformingLens);


};


struct LineLens :public Lens
{
	float lSemiMajorAxis, lSemiMinorAxis;
	float2 direction; //suppose normalized
	float ratio; //the ratio of lSemiMajorAxis and lSemiMinorAxis

	LineLens(int _r, float3 _c) : Lens(_c){
		lSemiMajorAxis = _r;
		ratio = 3.0f;
		lSemiMinorAxis = lSemiMajorAxis / ratio;

		direction = make_float2(1.0, 0.0);
		type = LENS_TYPE::TYPE_LINE;
		isConstructing = false;
	};

	bool PointInsideLens(int _x, int _y, float* mv, float* pj, int winW, int winH) {
		//sigmoid function: y=2*(1/(1+e^(-20*(x+1)))-0.5), x in [-1,0]
		//sigmoid function: y=2*(1/(1+e^(20*(x-1)))-0.5), x in [0,1]

		//dot product of (_x-x, _y-y) and direction
		float2 center = GetCenterScreenPos(mv, pj, winW, winH);
		float2 toPoint = make_float2(_x - center.x, _y - center.y);
		float disMajor = toPoint.x*direction.x + toPoint.y*direction.y;

		if (std::abs(disMajor) < lSemiMajorAxis) {
			float2 minorDirection = make_float2(-direction.y, direction.x);
			//dot product of (_x-x, _y-y) and minorDirection
			float disMinor = (_x - center.x)*minorDirection.x + (_y - center.y)*minorDirection.y;
			
			float disMajorRatio = disMajor / lSemiMajorAxis;
			float disSigmoid;
			if (disMajorRatio < 0){
				disSigmoid = 1 / (1 + exp(-40 * (disMajorRatio + 0.8))) ;
			}
			else {
				disSigmoid = 1 / (1 + exp(40 * (disMajorRatio - 0.8))) ;
			}

			if (std::abs(disMinor) < disSigmoid*lSemiMinorAxis)
				return true;
		}
		return false;
	}

	std::vector<float2> GetOuterContour(float* mv, float* pj, int winW, int winH) {
		std::vector<float2> ret;
		return ret;
	}

	std::vector<float2> GetContour(float* mv, float* pj, int winW, int winH) {
		//sigmoid function: y=2*(1/(1+e^(-20*(x+1)))-0.5), x in [-1,0]
		//sigmoid function: y=2*(1/(1+e^(20*(x-1)))-0.5), x in [0,1]
		float sigmoidCutOff = 0.4f; // assuming the sigmoid function value is constant when input is larger than sigmoidCutOff

		float2 minorDirection = make_float2(-direction.y, direction.x);

		std::vector<float2> ret;

		const int num_segments = 20;
		for (int ii = 0; ii < num_segments; ii++)
		{
			float tt = -1.0f + sigmoidCutOff*ii / num_segments;

			ret.push_back(GetCenterScreenPos(mv, pj, winW, winH) + tt*lSemiMajorAxis*direction
				+ (1 / (1 + exp(-40 * (tt + 0.8)))) *lSemiMinorAxis *minorDirection);//output vertex 
		}

		for (int ii = 0; ii < num_segments; ii++)
		{
			float tt = 1.0f - sigmoidCutOff + sigmoidCutOff*ii / num_segments;

			ret.push_back(GetCenterScreenPos(mv, pj, winW, winH) + tt*lSemiMajorAxis*direction
				+ (1 / (1 + exp(40 * (tt - 0.8)))) *lSemiMinorAxis *minorDirection);//output vertex 
		}

		
		for (int ii = 0; ii < num_segments; ii++)
		{
			float tt = 1.0f - sigmoidCutOff*ii / num_segments;

			ret.push_back(GetCenterScreenPos(mv, pj, winW, winH) + tt*lSemiMajorAxis*direction
				- (1 / (1 + exp(40 * (tt - 0.8)))) *lSemiMinorAxis *minorDirection);//output vertex 
		}

		for (int ii = 0; ii < num_segments; ii++)
		{
			float tt = -1.0f + sigmoidCutOff - sigmoidCutOff*ii / num_segments;

			ret.push_back(GetCenterScreenPos(mv, pj, winW, winH) + tt*lSemiMajorAxis*direction
				- (1 / (1 + exp(-40 * (tt + 0.8)))) *lSemiMinorAxis *minorDirection);//output vertex 
		}
		
		return ret;
	}

	std::vector<float2> GetCtrlPointsForRendering(float* mv, float* pj, int winW, int winH){
		std::vector<float2> res(0);
		return res;
	}
};

struct LineBLens :public Lens
{
	float lSemiMajorAxis, lSemiMinorAxis;
	float2 direction; //suppose normalized
	
	LineBLensInfo lineBLensInfo;

	float2 ctrlPoint1Abs, ctrlPoint2Abs; //only used during construction/transfornation. afterwards the center, direction, lSemiMajorAxis, and lSemiMinorAxis will be computed and recorded

	LineBLens(float3 _c, float _focusRatio = 0.5) : Lens(_c, _focusRatio){
		lSemiMajorAxis = 0;
		float ratio = 3.0f;
		lSemiMinorAxis = lSemiMajorAxis / ratio;
		direction = make_float2(1.0, 0.0);
		
		isConstructing = true;
		ctrlPoint1Abs = make_float2(0, 0);
		ctrlPoint2Abs = make_float2(0, 0);

		type = LENS_TYPE::TYPE_LINEB;

		UpdateLineBLensInfo();
	};

	void FinishConstructing(float* mv, float* pj, int winW, int winH);
	void UpdateInfo(float* mv, float* pj, int winW, int winH);

	bool PointInsideLens(int _x, int _y, float* mv, float* pj, int winW, int winH);

	std::vector<float2> GetOuterContour(float* mv, float* pj, int winW, int winH);

	std::vector<float2> GetContour(float* mv, float* pj, int winW, int winH);

	std::vector<float2> GetCtrlPointsForRendering(float* mv, float* pj, int winW, int winH);

	bool PointOnInnerBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) override;
	bool PointOnOuterBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) override;
	//bool PointOnCriticalPos(int _x, int _y, float* mv, float* pj, int winW, int winH) override;
	void ChangeLensSize(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH) override;
	void ChangefocusRatio(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH) override;
	//void ChangeDirection(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH) override;
	void UpdateLineBLensInfo();
};	




class CurveBLens :public Lens
{
	static const int numCtrlPointsLimit = 25;
	static const int distanceThr = 5;
	static const int distanceThrCount = 10;
	static const int refineIterationLimit = 2;

public:

	float width;
	float outerWidth;

	int numCtrlPoints;
	std::vector<float2> ctrlPoints; //used during constructing
	std::vector<float2> ctrlPointsAbs; //used during constructing to improve accuracy. will not be used after constructing

	std::vector<float2> BezierPoints; //sampled points on the curve for drawing
	std::vector<float2> BezierNormals; //sampled points on the curve for drawing

	std::vector<float2> subCtrlPointsPos; //may contain more number than numCtrlPoints, used for refining
	std::vector<float2> subCtrlPointsNeg; //may contain more number than numCtrlPoints, used for refining

	std::vector<float2> posOffsetCtrlPoints;
	std::vector<float2> negOffsetCtrlPoints;
	std::vector<float2> posOffsetBezierPoints;
	std::vector<float2> negOffsetBezierPoints;

	CurveBLensInfo curveBLensInfo;

	CurveBLens(int _w, float3 _c) : Lens(_c){

		isConstructing = true;

		width = _w;
		focusRatio = 0.5;
		outerWidth = width / focusRatio;
		numCtrlPoints = 0;
		type = LENS_TYPE::TYPE_CURVEB;
	};

	void AddCtrlPoint(int _x, int _y){

		if (numCtrlPoints < numCtrlPointsLimit) {

			//check if the candidate point is not too close to previous points
			int tt = max(0, numCtrlPoints - distanceThrCount);
			bool notFoundTooClose = true;
			for (int i = numCtrlPoints - 1; i >= tt && notFoundTooClose; i--){
				if (length(ctrlPointsAbs[i] - make_float2(_x, _y)) < distanceThr)
					notFoundTooClose = false;
			}
			if (notFoundTooClose) {
				ctrlPointsAbs.push_back(make_float2(_x, _y));
				numCtrlPoints++;
			}
		}
	}

	

	bool PointInsideLens(int _x, int _y, float* mv, float* pj, int winW, int winH);

	void FinishConstructing(float* mv, float* pj, int winW, int winH);
	std::vector<float2> GetContour(float* mv, float* pj, int winW, int winH);
	std::vector<float2> GetOuterContour(float* mv, float* pj, int winW, int winH);
	std::vector<float2> GetOuterContourold();
	std::vector<float2> GetCtrlPointsForRendering(float* mv, float* pj, int winW, int winH);
	std::vector<float2> GetCenterLineForRendering(float* mv, float* pj, int winW, int winH);

	std::vector<float2> removeSelfIntersection(std::vector<float2> p, bool isDuplicating);
	bool adjustOffset();
	int refinedRoundPos;
	int refinedRoundNeg;
	void RefineLensBoundary();

	void offsetControlPointsPos();
	void offsetControlPointsNeg();

	void computeBoundaryPos();
	void computeBoundaryNeg();
	void computeRenderingBoundaryPos(std::vector<float2> &ret, int bezierSampleAccuracyRate);
	void computeRenderingBoundaryNeg(std::vector<float2> &ret, int bezierSampleAccuracyRate);

	bool ccw(float2 A, float2 B, float2 C) //counter clock wise
	{
		return (C.y - A.y)*(B.x - A.x) >(B.y - A.y)*(C.x - A.x);
	}

	bool intersect(float2 A, float2 B, float2 C, float2 D) //whether segment AB and segmnent CD is intersected
	{
		float lengthThr = 0.00001;
		if (length(A - B) < lengthThr || length(C - D) < lengthThr || length(A - C) < lengthThr ||
			length(A - D) < lengthThr || length(C - B) < lengthThr || length(D - B) < lengthThr)
			return false;
		return (ccw(A, C, D) != ccw(B, C, D) && ccw(A, B, C) != ccw(A, B, D));
	}

	void intersectPoint(float2 p1, float2 p2, float2 p3, float2 p4, float2 &ret) //get the intersected point of line p1p2 and p3p4. return the result in P. if parallel, return the mid point of C and D
		//source: http://stackoverflow.com/questions/4543506/algorithm-for-intersection-of-2-lines
	{
		//line p1p2: (x-p1.x)/(p2.x-p1.x) = (y-p1.y)/(p2.y-p1.y)
		//change to form Ax + By = C
		float A1 = p2.y - p1.y, B1 = -(p2.x - p1.x), C1 = p1.x*(p2.y - p1.y) - p1.y*(p2.x - p1.x);
		float A2 = p4.y - p3.y, B2 = -(p4.x - p3.x), C2 = p3.x*(p4.y - p3.y) - p3.y*(p4.x - p3.x);
		float delta = A1*B2 - A2*B1;
		if (std::abs((long)delta) < 0.000001){
			ret = (p2 + p3) / 2;
		}
		else{
			ret = make_float2((B2*C1 - B1*C2) / delta, (A1*C2 - A2*C1) / delta);
		}
	}

	void UpdateTransferredData()
	{
		//!!! may need to be more sophisticate
		//curveLensCtrlPoints.focusRatio = focusRatio;
	}

	std::vector<float2> BezierOneSubdivide(std::vector<float2> p, std::vector<float2> poly1, std::vector<float2> poly2, float u);
	std::vector<float2> BezierSubdivide(std::vector<float2> p, int m, float u);
	std::vector<float2> BezierSmaple(std::vector<float2> p, int bezierSampleAccuracyRate = 1);
	std::vector<float2> BezierSmaple(std::vector<float2> p, std::vector<float> us);//for computing the tangent

};

#endif