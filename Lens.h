#ifndef LENS_H
#define LENS_H
#include <vector_types.h>
#include <helper_math.h>
#include <vector>

#include <iostream>
using namespace std;

enum LENS_TYPE{
	TYPE_CIRCLE,
	TYPE_LINE,
	TYPE_POLYLINE,
	TYPE_CURVE,
	TYPE_CURVEB
};

struct PolyLineLensCtrlPoints
{
	int numCtrlPoints;
	float2 ctrlPoints[20];
	float2 dirs[19];
	float2 angleBisectors[20];
};

struct CurveLensCtrlPoints
{
	int numCtrlPoints;
	float2 ctrlPoints[200];
	int numKeyPoints;
	float2 keyPoints[200];
	int keyPointIds[200];
	float2 normals[200];
	float focusRatio;//ratio of focus region and transition region
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
	virtual std::vector<float2> GetExtraLensRendering() = 0;
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

	std::vector<float2> GetExtraLensRendering(){
		vector<float2> res(0);
		return res;
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

	std::vector<float2> GetExtraLensRendering(){
		vector<float2> res(0);
		return res;
	}
};

struct PolyLineLens :public Lens
{
	int width;
	int numCtrlPoints;
	//!note!  positions relative to the center

	float2 direction;
	float lSemiMajor, lSemiMinor;

	std::vector<float2> ctrlPoints;
	std::vector<float2> dirs;
	std::vector<float2> angleBisectors;

	//used for transfering data to GPU
	PolyLineLensCtrlPoints polyLineLensCtrlPoints;

	bool isConstructing;

	void FinishConstructing(){
		if (numCtrlPoints >= 2){
			computePCA();

			isConstructing = false;
		}
	}

	PolyLineLens(int _x, int _y, int _w, float3 _c) : Lens(_x, _y, _c){

		isConstructing = true;

		width = _w;

		numCtrlPoints = 0;
		type = LENS_TYPE::TYPE_POLYLINE;
	};

	std::vector<float2> GetOuterContour() {
		std::vector<float2> ret;
		return ret;
	}

	void computePCA(){

		if (numCtrlPoints == 0)
		{
		}
		else if (numCtrlPoints == 1)
		{
			lSemiMajor = 10;
			lSemiMinor = 10;
			direction = normalize(make_float2(1.0, 0.0));

		}
		else{
			//compute PCA
			//X is matrix formed by row vectors of points, i.e., X=[ctrlPoints[0].x, ctrlPoints[0].y;ctrlPoints[1].x, ctrlPoints[1].y;...]
			//get the eignevector of A = X'X
			float A11 = 0, A12 = 0, A21 = 0, A22 = 0;
			for (int ii = 0; ii < numCtrlPoints; ii++) {
				A11 += ctrlPoints[ii].x * ctrlPoints[ii].x;
				A22 += ctrlPoints[ii].y * ctrlPoints[ii].y;
				A12 += ctrlPoints[ii].x * ctrlPoints[ii].y;
				A21 += ctrlPoints[ii].y * ctrlPoints[ii].x;
			}
			//Av=lv -> (A-lI)v=0 -> f(l) = |A-lI| = 0 -> (A11-l)*(A22-l)-A12*A21 = 0
			//-> l^2 - (A11+A22)*l + A11*A22-A12*A21 = 0;
			float equa = 1, equb = -(A11 + A22), equc = A11*A22 - A12*A21;
			float lembda1 = (-equb + sqrt(equb*equb - 4 * equa * equc)) / 2.0 / equa;
			//Av=lv -> (A-lI)v=0 -> (A11-l, A12) .* (v1, v2) = 0
			direction = normalize(make_float2(-A12, A11 - lembda1));

			float2 minorDirection = make_float2(-direction.y, direction.x);

			float maxMajor = -9999, maxMinor = -9999, minMajor = 9999, minMinor = 9999;
			for (int ii = 0; ii < numCtrlPoints; ii++) {
				//(ctrlPoints[ii].x - x, ctrlPoints[ii].y - y) dot product direction (and minorDirection)
				float disMajorii = (ctrlPoints[ii].x)*direction.x + (ctrlPoints[ii].y)*direction.y;
				float disMinorii = (ctrlPoints[ii].x)*minorDirection.x + (ctrlPoints[ii].y)*minorDirection.y;
				if (disMajorii > maxMajor)
					maxMajor = disMajorii;
				if (disMajorii < minMajor)
					minMajor = disMajorii;
				if (disMinorii > maxMinor)
					maxMinor = disMinorii;
				if (disMinorii < minMinor)
					minMinor = disMinorii;
			}

			lSemiMajor = max(abs(maxMajor), abs(minMajor));
			lSemiMinor = max(abs(maxMinor), abs(minMinor));

			lSemiMajor = lSemiMajor + max(lSemiMajor*0.1, 10.0);
			lSemiMinor = lSemiMinor + max(lSemiMinor*0.1, 10.0);
		}
	}
	void AddCtrlPoint(int _x, int _y){

		if (numCtrlPoints == 0){
			x = _x;
			y = _y;
			ctrlPoints.push_back(make_float2(0, 0));
			numCtrlPoints = 1;
		}
		else {
			//do the process due to the relative position

			float sumx = x*numCtrlPoints, sumy = y*numCtrlPoints;
			sumx += _x, sumy += _y;  //sum of absolute position
			float newx = sumx / (numCtrlPoints + 1), newy = sumy / (numCtrlPoints + 1);

			for (int ii = 0; ii < numCtrlPoints; ii++) {
				ctrlPoints[ii].x = ctrlPoints[ii].x + x - newx;
				ctrlPoints[ii].y = ctrlPoints[ii].y + y - newy;
			}

			ctrlPoints.push_back(make_float2(_x - newx, _y - newy));
			numCtrlPoints++;

			x = newx;
			y = newy;
		}

		dirs.resize(numCtrlPoints - 1);
		for (int ii = 0; ii < numCtrlPoints - 1; ii++){
			dirs[ii] = normalize(ctrlPoints[ii + 1] - ctrlPoints[ii]);
		}

		if (numCtrlPoints >= 2) {
			angleBisectors.resize(numCtrlPoints);
			angleBisectors[0] = make_float2(-dirs[0].y, dirs[0].x);
			for (int ii = 1; ii < numCtrlPoints - 1; ii++){
				float2 perpenAngleBosector = normalize(dirs[ii - 1] + dirs[ii]);
				angleBisectors[ii] = make_float2(-perpenAngleBosector.y, perpenAngleBosector.x);
			}
			angleBisectors[numCtrlPoints - 1] = make_float2(-dirs[numCtrlPoints - 2].y, dirs[numCtrlPoints - 2].x);
		}


		//note!!! currently limit the maximum amount of control points as 20
		if (numCtrlPoints > 20)
			polyLineLensCtrlPoints.numCtrlPoints = 20;
		else
			polyLineLensCtrlPoints.numCtrlPoints = numCtrlPoints;

		for (int ii = 0; ii < polyLineLensCtrlPoints.numCtrlPoints - 1; ii++) {
			polyLineLensCtrlPoints.ctrlPoints[ii] = ctrlPoints[ii];
			polyLineLensCtrlPoints.dirs[ii] = dirs[ii];
			polyLineLensCtrlPoints.angleBisectors[ii] = angleBisectors[ii];
		}
		polyLineLensCtrlPoints.ctrlPoints[polyLineLensCtrlPoints.numCtrlPoints - 1]
			= ctrlPoints[polyLineLensCtrlPoints.numCtrlPoints - 1];
		if (numCtrlPoints >= 2) {
			polyLineLensCtrlPoints.angleBisectors[polyLineLensCtrlPoints.numCtrlPoints - 1]
				= angleBisectors[polyLineLensCtrlPoints.numCtrlPoints - 1];
		}
	}

	bool PointInsideLens(int _x, int _y)
	{
		bool segmentNotFound = true;
		for (int ii = 0; ii < numCtrlPoints - 1 && segmentNotFound; ii++) {
			float2 center = make_float2(x, y);
			float2 screenPos = make_float2(_x, _y);
			float2 toPoint = screenPos - (center + ctrlPoints[ii]);
			float2 dir = dirs[ii];
			float2 minorDir = make_float2(-dir.y, dir.x);
			float disMinor = toPoint.x*minorDir.x + toPoint.y*minorDir.y;
			if (abs(disMinor) < width)	{
				float2 ctrlPointAbsolute1 = center + ctrlPoints[ii];
				float2 ctrlPointAbsolute2 = center + ctrlPoints[ii + 1];

				//first check if screenPos and ctrlPointAbsolute2 are at the same side of Line (ctrlPointAbsolute1, angleBisectors[ii])
				//then check if screenPos and ctrlPointAbsolute1 are at the same side of Line (ctrlPointAbsolute2, angleBisectors[ii+1])
				if (((screenPos.x - ctrlPointAbsolute1.x)*angleBisectors[ii].y - (screenPos.y - ctrlPointAbsolute1.y)*angleBisectors[ii].x)
					*((ctrlPointAbsolute2.x - ctrlPointAbsolute1.x)*angleBisectors[ii].y - (ctrlPointAbsolute2.y - ctrlPointAbsolute1.y)*angleBisectors[ii].x)
					>= 0) {
					if (((screenPos.x - ctrlPointAbsolute2.x)*angleBisectors[ii + 1].y - (screenPos.y - ctrlPointAbsolute2.y)*angleBisectors[ii + 1].x)
						*((ctrlPointAbsolute1.x - ctrlPointAbsolute2.x)*angleBisectors[ii + 1].y - (ctrlPointAbsolute1.y - ctrlPointAbsolute2.y)*angleBisectors[ii + 1].x)
						>= 0) {

						segmentNotFound = false;
					}
				}

			}
		}

		return !segmentNotFound;
	}

	std::vector<float2> GetContour(){
		std::vector<float2> ret;
		if (numCtrlPoints == 1){
			float2 center = make_float2(x, y);

			float2 rightUp = normalize(make_float2(1.0, 1.0));
			float2 rightDown = normalize(make_float2(1.0, -1.0));

			ret.push_back(center + rightUp*width);
			ret.push_back(center + rightDown*width);
			ret.push_back(center - rightUp*width);
			ret.push_back(center - rightDown*width);

		}
		else if (numCtrlPoints >= 2) {
			ret.resize(2 * numCtrlPoints);
			float2 center = make_float2(x, y);
			for (int ii = 0; ii < numCtrlPoints; ii++) {
				float2 dir;
				if (ii == numCtrlPoints - 1)
					dir = dirs[numCtrlPoints - 2];
				else
					dir = dirs[ii];
				float sinValue = dir.x*angleBisectors[ii].y - dir.y*angleBisectors[ii].x;
				ret[ii] = center + ctrlPoints[ii] + angleBisectors[ii] * width / sinValue;
				ret[2 * numCtrlPoints - 1 - ii] = center + ctrlPoints[ii] - angleBisectors[ii] * width / sinValue;
			}
		}
		return ret;
	}

	std::vector<float2> GetExtraLensRendering(){
		std::vector<float2> ret;

		for (int ii = 0; ii < numCtrlPoints; ii++) {
			ret.push_back(make_float2(ctrlPoints[ii].x + x, ctrlPoints[ii].y + y));
		}

		return ret;
	}

};




struct CurveLens :public Lens
{
#define numCtrlPointsLimit 500
#define distanceThr 1
#define distanceThrCount 10

public:

	int width;
	int numCtrlPoints;
	std::vector<float2> ctrlPoints;
	std::vector<float2> ctrlPointsAbs; //used during constructing to improve accuracy. will not be used after constructing

	bool isConstructing;

	CurveLensCtrlPoints curveLensCtrlPoints;

	CurveLens(int _x, int _y, int _w, float3 _c) : Lens(_x, _y, _c){

		isConstructing = true;

		width = _w;
		focusRatio = 0.5;
		numCtrlPoints = 0;
		type = LENS_TYPE::TYPE_CURVE;
	};

	void AddCtrlPoint(int _x, int _y){

		if (numCtrlPoints < numCtrlPointsLimit) {

			//first check if the candidate point is not too close to previous points
			int tt = max(0, numCtrlPoints - distanceThrCount);
			bool notFoundTooClose = true;
			for (int i = numCtrlPoints - 1; i >= tt; i--){
				if (length(ctrlPointsAbs[i] - make_float2(_x, _y)) < distanceThr)
					notFoundTooClose = false;
			}
			if (notFoundTooClose) {
				ctrlPointsAbs.push_back(make_float2(_x, _y));
				numCtrlPoints++;
			}
		}
	}

	void FinishConstructing(){
		if (numCtrlPoints >= 3){
			isConstructing = false;

			float sumx = 0, sumy = 0;
			for (int ii = 0; ii < numCtrlPoints; ii++) {
				sumx += ctrlPointsAbs[ii].x, sumy += ctrlPointsAbs[ii].y;  //sum of absolute position
			}
			x = sumx / numCtrlPoints, y = sumy / numCtrlPoints;
			ctrlPoints.resize(numCtrlPoints);
			for (int ii = 0; ii < numCtrlPoints; ii++) {
				ctrlPoints[ii].x = ctrlPointsAbs[ii].x - x;
				ctrlPoints[ii].y = ctrlPointsAbs[ii].y - y;
			}

			//compute curveLensCtrlPoints
			curveLensCtrlPoints.focusRatio = focusRatio;
			curveLensCtrlPoints.numCtrlPoints = numCtrlPoints;
			for (int i = 0; i < numCtrlPoints; i++){
				curveLensCtrlPoints.ctrlPoints[i] = ctrlPoints[i];
			}

			std::vector<float2> sidePointsPos, sidePointsNeg;

			int numKeyPoints = 0;

			float2 center = make_float2(x, y);

			int lastValidID = 0;
			for (int ii = 0; ii < numCtrlPoints; ii++) {
				float2 dir; //tangent
				if (ii == numCtrlPoints - 1)
					dir = normalize(ctrlPoints[numCtrlPoints - 1] - ctrlPoints[numCtrlPoints - 2]);
				else if (ii == 0)
					dir = normalize(ctrlPoints[1] - ctrlPoints[0]);
				else
					dir = normalize((ctrlPoints[ii + 1] - ctrlPoints[ii - 1]) / 2);

				float2 normal = make_float2(-dir.y, dir.x);

				if (ii == 0){
					sidePointsPos.push_back(center + ctrlPoints[0] + normal*width);
					sidePointsNeg.push_back(center + ctrlPoints[0] - normal*width);

					curveLensCtrlPoints.keyPoints[numKeyPoints] = ctrlPoints[ii];
					lastValidID = 0;
					curveLensCtrlPoints.keyPointIds[numKeyPoints] = lastValidID;
					curveLensCtrlPoints.normals[numKeyPoints] = normal;
					numKeyPoints++;
				}
				//else if (ii == numCtrlPoints - 1){

				//}
				else{
					float2 candiPos = center + ctrlPoints[ii] + normal*width;
					float2 candiNeg = center + ctrlPoints[ii] - normal*width;
					float2 candiPosTransitionRegion = center + ctrlPoints[ii] + normal*width / focusRatio;
					float2 candiNegTransitionRegion = center + ctrlPoints[ii] - normal*width / focusRatio;

					if (!intersect(center + ctrlPoints[lastValidID], 2 * sidePointsPos[numKeyPoints - 1] - (center + ctrlPoints[lastValidID]),
						center + ctrlPoints[ii], candiPosTransitionRegion)
					 && !intersect(center + ctrlPoints[lastValidID], 2 * sidePointsNeg[numKeyPoints - 1] - (center + ctrlPoints[lastValidID]),
						center + ctrlPoints[ii], candiNegTransitionRegion)){
						sidePointsPos.push_back(candiPos);
						sidePointsNeg.push_back(candiNeg);
						curveLensCtrlPoints.keyPoints[numKeyPoints] = ctrlPoints[ii];
						lastValidID = ii;
						curveLensCtrlPoints.keyPointIds[numKeyPoints] = lastValidID;
						curveLensCtrlPoints.normals[numKeyPoints] = normal;
						numKeyPoints++;
					}
				}
			}

			curveLensCtrlPoints.numKeyPoints = numKeyPoints;
		}
	}

	bool PointInsideLens(int _x, int _y)
	{
		if (isConstructing)
			return true;

		float2 screenPos = make_float2(_x, _y);
		int numCtrlPoints = curveLensCtrlPoints.numCtrlPoints;
		float2* ctrlPoints = curveLensCtrlPoints.ctrlPoints;

		float2* normals = curveLensCtrlPoints.normals;
		int numKeyPoints = curveLensCtrlPoints.numKeyPoints;
		float2* keyPoints = curveLensCtrlPoints.keyPoints;
		int* keyPointIds = curveLensCtrlPoints.keyPointIds;

		bool segmentNotFound = true;
		int keySegmentId = -1;
		for (int ii = 0; ii < numKeyPoints - 1 && segmentNotFound; ii++) {
			float2 center = make_float2(x, y);
			float2 toPoint = screenPos - (center + keyPoints[ii]);
			float2 dir = normalize(keyPoints[ii + 1] - keyPoints[ii]);
			float2 minorDir = make_float2(-dir.y, dir.x);
			float disMinor = toPoint.x*minorDir.x + toPoint.y*minorDir.y;
			if (abs(disMinor) < width)	{
				float2 ctrlPointAbsolute1 = center + keyPoints[ii];
				float2 ctrlPointAbsolute2 = center + keyPoints[ii + 1];

				//first check if screenPos and ctrlPointAbsolute2 are at the same side of Line (ctrlPointAbsolute1, normals[ii])
				//then check if screenPos and ctrlPointAbsolute1 are at the same side of Line (ctrlPointAbsolute2, normals[ii+1])

				if (((screenPos.x - ctrlPointAbsolute1.x)*normals[ii].y - (screenPos.y - ctrlPointAbsolute1.y)*normals[ii].x)
					*((ctrlPointAbsolute2.x - ctrlPointAbsolute1.x)*normals[ii].y - (ctrlPointAbsolute2.y - ctrlPointAbsolute1.y)*normals[ii].x)
					>= 0) {
					if (((screenPos.x - ctrlPointAbsolute2.x)*normals[ii + 1].y - (screenPos.y - ctrlPointAbsolute2.y)*normals[ii + 1].x)
						*((ctrlPointAbsolute1.x - ctrlPointAbsolute2.x)*normals[ii + 1].y - (ctrlPointAbsolute1.y - ctrlPointAbsolute2.y)*normals[ii + 1].x)
						>= 0) {
						segmentNotFound = false;
					}
				}
			}
		}
		return !segmentNotFound;
	}

	std::vector<float2> GetContour(){
		std::vector<float2> ret;

		if (!isConstructing && numCtrlPoints >= 3) {
			std::vector<float2> sidePointsPos, sidePointsNeg;

			float2 center = make_float2(x, y);

			int numKeyPoints = curveLensCtrlPoints.numKeyPoints;
			float2 * keyPoints = curveLensCtrlPoints.keyPoints;
			float2 * normals = curveLensCtrlPoints.normals;

			ret.resize(2 * numKeyPoints);
			for (int jj = 0; jj < numKeyPoints; jj++){
				ret[jj] = center + keyPoints[jj] + normals[jj] * width;
				ret[2 * numKeyPoints - 1 - jj] = center + keyPoints[jj] - normals[jj] * width;
			}
		}

		return ret;
	}

	std::vector<float2> GetOuterContour() {
		std::vector<float2> ret;

		if (!isConstructing && numCtrlPoints >= 3) {
			std::vector<float2> sidePointsPos, sidePointsNeg;

			float2 center = make_float2(x, y);

			int numKeyPoints = curveLensCtrlPoints.numKeyPoints;
			float2 * keyPoints = curveLensCtrlPoints.keyPoints;
			float2 * normals = curveLensCtrlPoints.normals;

			ret.resize(2 * numKeyPoints);
			for (int jj = 0; jj < numKeyPoints; jj++){
				ret[jj] = center + keyPoints[jj] + normals[jj] * width / focusRatio;
				ret[2 * numKeyPoints - 1 - jj] = center + keyPoints[jj] - normals[jj] * width / focusRatio;;
			}
		}

		return ret;
	}

	std::vector<float2> GetExtraLensRendering(){
		std::vector<float2> ret;
		if (isConstructing){
			for (int ii = 0; ii < numCtrlPoints; ii++) {
				ret.push_back(make_float2(ctrlPointsAbs[ii].x, ctrlPointsAbs[ii].y));
			}
		}
		else{
			for (int ii = 0; ii < numCtrlPoints; ii++) {
				ret.push_back(make_float2(ctrlPoints[ii].x + x, ctrlPoints[ii].y + y));
			}
		}
		return ret;
	}

	bool ccw(float2 A, float2 B, float2 C) //counter clock wise
	{
		return (C.y - A.y)*(B.x - A.x) >(B.y - A.y)*(C.x - A.x);
	}

	bool intersect(float2 A, float2 B, float2 C, float2 D)
	{
		return (ccw(A, C, D) != ccw(B, C, D) && ccw(A, B, C) != ccw(A, B, D));
	}

	void UpdateTransferredData()
	{
		//!!! may need to be more sophisticate
		curveLensCtrlPoints.focusRatio = focusRatio;
	}
};


class CurveBLens :public Lens
{
#define numCtrlPointsLimit 500
#define distanceThr 1
#define distanceThrCount 10

public:

	int width;
	int numCtrlPoints;
	std::vector<float2> ctrlPoints;
	std::vector<float2> ctrlPointsAbs; //used during constructing to improve accuracy. will not be used after constructing


	std::vector<float2> BezierCtrlPoints; //may contain more number than numCtrlPoints

	std::vector<float2> BezierPosOffset;
	std::vector<float2> BezierNegOffset;

	bool isConstructing;

	CurveLensCtrlPoints curveLensCtrlPoints;

	CurveBLens(int _x, int _y, int _w, float3 _c) : Lens(_x, _y, _c){

		isConstructing = true;

		width = _w;
		focusRatio = 0.5;
		numCtrlPoints = 0;
		type = LENS_TYPE::TYPE_CURVEB;
	};

	void AddCtrlPoint(int _x, int _y){

		if (numCtrlPoints < numCtrlPointsLimit) {

			//first check if the candidate point is not too close to previous points
			int tt = max(0, numCtrlPoints - distanceThrCount);
			bool notFoundTooClose = true;
			for (int i = numCtrlPoints - 1; i >= tt; i--){
				if (length(ctrlPointsAbs[i] - make_float2(_x, _y)) < distanceThr)
					notFoundTooClose = false;
			}
			if (notFoundTooClose) {
				ctrlPointsAbs.push_back(make_float2(_x, _y));
				numCtrlPoints++;
			}
		}
	}

	

	bool PointInsideLens(int _x, int _y)
	{
		if (isConstructing)
			return true;
		return true;

		float2 screenPos = make_float2(_x, _y);
		int numCtrlPoints = curveLensCtrlPoints.numCtrlPoints;
		float2* ctrlPoints = curveLensCtrlPoints.ctrlPoints;

		float2* normals = curveLensCtrlPoints.normals;
		int numKeyPoints = curveLensCtrlPoints.numKeyPoints;
		float2* keyPoints = curveLensCtrlPoints.keyPoints;
		int* keyPointIds = curveLensCtrlPoints.keyPointIds;

		bool segmentNotFound = true;
		int keySegmentId = -1;
		for (int ii = 0; ii < numKeyPoints - 1 && segmentNotFound; ii++) {
			float2 center = make_float2(x, y);
			float2 toPoint = screenPos - (center + keyPoints[ii]);
			float2 dir = normalize(keyPoints[ii + 1] - keyPoints[ii]);
			float2 minorDir = make_float2(-dir.y, dir.x);
			float disMinor = toPoint.x*minorDir.x + toPoint.y*minorDir.y;
			if (abs(disMinor) < width)	{
				float2 ctrlPointAbsolute1 = center + keyPoints[ii];
				float2 ctrlPointAbsolute2 = center + keyPoints[ii + 1];

				//first check if screenPos and ctrlPointAbsolute2 are at the same side of Line (ctrlPointAbsolute1, normals[ii])
				//then check if screenPos and ctrlPointAbsolute1 are at the same side of Line (ctrlPointAbsolute2, normals[ii+1])

				if (((screenPos.x - ctrlPointAbsolute1.x)*normals[ii].y - (screenPos.y - ctrlPointAbsolute1.y)*normals[ii].x)
					*((ctrlPointAbsolute2.x - ctrlPointAbsolute1.x)*normals[ii].y - (ctrlPointAbsolute2.y - ctrlPointAbsolute1.y)*normals[ii].x)
					>= 0) {
					if (((screenPos.x - ctrlPointAbsolute2.x)*normals[ii + 1].y - (screenPos.y - ctrlPointAbsolute2.y)*normals[ii + 1].x)
						*((ctrlPointAbsolute1.x - ctrlPointAbsolute2.x)*normals[ii + 1].y - (ctrlPointAbsolute1.y - ctrlPointAbsolute2.y)*normals[ii + 1].x)
						>= 0) {
						segmentNotFound = false;
					}
				}
			}
		}
		return !segmentNotFound;
	}

	void FinishConstructing();
	std::vector<float2> GetContour();
	std::vector<float2> GetOuterContour();
	std::vector<float2> GetOuterContourold();
	std::vector<float2> GetExtraLensRendering();
	std::vector<float2> GetExtraLensRendering2();


	void RefineLensCenter();

	bool ccw(float2 A, float2 B, float2 C) //counter clock wise
	{
		return (C.y - A.y)*(B.x - A.x) >(B.y - A.y)*(C.x - A.x);
	}

	bool intersect(float2 A, float2 B, float2 C, float2 D) //whether segment AB and segmnent CD is intersected
	{
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
		if (abs(delta) < 0.000001){
			ret = (p2 + p3) / 2;
		}
		else{
			ret = make_float2((B2*C1 - B1*C2) / delta, (A1*C2 - A2*C1) / delta);
		}
	}

	void UpdateTransferredData()
	{
		//!!! may need to be more sophisticate
		curveLensCtrlPoints.focusRatio = focusRatio;
	}



	vector<float2> BezierSmaple(vector<float2> p)
	{
		vector<float2> res;
		if (p.size() >= 2){
#define bezierSampleAccuracyRate 1 
			int n = p.size() - 1;

			double *combinationValue = new double[n + 1];
			for (int i = 0; i <= n/2; i++){
				double cc = 1; //compute n!/i!/(n-i)! = ((n-i+1)*...*n)/(1*2*...*i)
				for (int j = i; j >=1; j--){
					cc = cc*(n + j - i) / j;
				}
				combinationValue[i] = cc;
			}
			for (int i = n/2+1; i <= n; i++){
				combinationValue[i] = combinationValue[n-i];
			}

			for (int ui = 0; ui <= n*bezierSampleAccuracyRate; ui++){
				float u = ui*1.0 / (n*bezierSampleAccuracyRate);
				float2 pu = make_float2(0, 0);
				for (int j = 0; j <= n; j++){
					pu = pu + combinationValue[j] * pow(u, j) * pow(1 - u, n - j) * p[j];
				}
				res.push_back(pu);
			}

			delete combinationValue;
		}
		return res;
	}

};

#endif