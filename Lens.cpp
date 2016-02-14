#include "Lens.h"
#include "TransformFunc.h"
#include <vector_functions.h>
#include <helper_math.h>
#include <iostream>

float Lens::GetClipDepth(float* mv, float* pj)
{
	return Object2Clip(GetCenter(), mv, pj).z;
}

float4 Lens::GetCenter() { return make_float4(c.x, c.y, c.z, 1.0f); }

void Lens::ChangeClipDepth(int v, float* mv, float* pj)
{
	matrix4x4 invModelview, invProjection;
	invertMatrix(mv, &invModelview.v[0].x);
	invertMatrix(pj, &invProjection.v[0].x);
	float4 cenClip = Object2Clip(GetCenter(), mv, pj);
	//std::cout << cenClip.z << std::endl;
	float4 cenShiftClip = cenClip + make_float4(0, 0, -0.01, 0);
	float4 cenShiftObj = Clip2ObjectGlobal(cenShiftClip, &invModelview.v[0].x, &invProjection.v[0].x);
	float4 dir4 = cenShiftObj - GetCenter();// make_float3(dir_object.x, dir_object.y, dir_object.z);
	float3 dir3 = make_float3(dir4.x, dir4.y, dir4.z);
	dir3 = dir3 * (1.0f / length(dir3)) * v * (-0.05);
	SetCenter(make_float3(
		c.x + dir3.x,
		c.y + dir3.y,
		c.z + dir3.z));
}





void CurveBLens::FinishConstructing(){
	if (numCtrlPoints >= 3){

		cout << "numCtrlPoints: " << numCtrlPoints << endl;
		//remove self intersection
		ctrlPointsAbs = removeSelfIntersection(ctrlPointsAbs, false);
		numCtrlPoints = ctrlPointsAbs.size();

		//compute center and relative positions
		float sumx = 0, sumy = 0;
		for (int ii = 0; ii < numCtrlPoints; ii++) {
			sumx += ctrlPointsAbs[ii].x, sumy += ctrlPointsAbs[ii].y;  //sum of absolute position
		}
		x = sumx / numCtrlPoints, y = sumy / numCtrlPoints;
		float2 center = make_float2(x, y);
		ctrlPoints.resize(numCtrlPoints);
		for (int ii = 0; ii < numCtrlPoints; ii++) {
			ctrlPoints[ii] = ctrlPointsAbs[ii] - center;
		}

		//optional: refine the control points shape and reduce the number, by the Bezier Curve
		if (0){
			vector<float2> BezierSmapleOri = BezierSmaple(ctrlPoints);
			numCtrlPoints = numCtrlPoints / 2;
			ctrlPoints.resize(numCtrlPoints);
			for (int ii = 0; ii < numCtrlPoints; ii++) {
				ctrlPoints[ii] = BezierSmapleOri[ii * 2];
			}
		}

		//use another array to store the control points, since it might be changed later to refine the offset
		rationalCtrlPoints = ctrlPoints;

		//compute the BezierPoints used to draw the curve
		BezierPoints = BezierSmaple(ctrlPoints);

		//compute posOffsetCtrlPoints
		int n = rationalCtrlPoints.size();
		float2 normal;
		float2 dir;

		dir = normalize(rationalCtrlPoints[1] - rationalCtrlPoints[0]);
		normal = make_float2(-dir.y, dir.x);
		posOffsetCtrlPoints.push_back(rationalCtrlPoints[0] + normal*width);
		negOffsetCtrlPoints.push_back(rationalCtrlPoints[0] - normal*width);

		for (int i = 1; i < n - 1; i++){
			dir = normalize(rationalCtrlPoints[i] - rationalCtrlPoints[i - 1]);
			normal = make_float2(-dir.y, dir.x);
			float2 dir2 = normalize(rationalCtrlPoints[i + 1] - rationalCtrlPoints[i]);
			float2 normal2 = make_float2(-dir.y, dir.x);
			float2 posCtrlPoint;
			intersectPoint(rationalCtrlPoints[i - 1] + normal*width, rationalCtrlPoints[i] + normal*width, rationalCtrlPoints[i] + normal2*width, rationalCtrlPoints[i + 1] + normal2*width, posCtrlPoint);
			posOffsetCtrlPoints.push_back(posCtrlPoint);

			float2 negCtrlPoint;
			intersectPoint(rationalCtrlPoints[i - 1] - normal*width, rationalCtrlPoints[i] - normal*width, rationalCtrlPoints[i] - normal2*width, rationalCtrlPoints[i + 1] - normal2*width, negCtrlPoint);
			negOffsetCtrlPoints.push_back(negCtrlPoint);
		}

		dir = normalize(rationalCtrlPoints[n - 1] - rationalCtrlPoints[n - 2]);
		normal = make_float2(-dir.y, dir.x);
		posOffsetCtrlPoints.push_back(rationalCtrlPoints[n - 1] + normal*width);
		negOffsetCtrlPoints.push_back(rationalCtrlPoints[n - 1] - normal*width);
		
		posOffsetBezierPoints = BezierSmaple(posOffsetCtrlPoints);
		negOffsetBezierPoints = BezierSmaple(negOffsetCtrlPoints);


		////do it later: compute curveLensCtrlPoints
		if (0){
			curveLensCtrlPoints.focusRatio = focusRatio;
			curveLensCtrlPoints.numCtrlPoints = numCtrlPoints;
			for (int i = 0; i < numCtrlPoints; i++){
				curveLensCtrlPoints.ctrlPoints[i] = ctrlPoints[i];
			}
		}


		isConstructing = false;
	}
}



std::vector<float2> CurveBLens::GetContour(){
	std::vector<float2> ret;
	if (posOffsetBezierPoints.size()>2){
	//if (!isConstructing && numCtrlPoints >= 3) {
		float2 center = make_float2(x, y);
		int n = posOffsetBezierPoints.size();
		for (int ii = 0; ii < n; ii++){
			ret.push_back(posOffsetBezierPoints[ii] + center);
		}
		std::vector<float2> rettemp;
		int m = negOffsetBezierPoints.size();
		for (int ii = 0; ii < m; ii++){
			rettemp.push_back(negOffsetBezierPoints[ii] + center);
		}
		
		std::reverse(rettemp.begin(), rettemp.end());
		ret.insert(ret.end(), rettemp.begin(), rettemp.end());

	}
	return ret;
}


std::vector<float2> CurveBLens::GetOuterContour() {
	std::vector<float2> ret; return ret;

	if (!isConstructing && numCtrlPoints >= 3) {
		std::vector<float2> sidePointsPos, sidePointsNeg;
		float2 center = make_float2(x, y);

		for (int ii = 0; ii < numCtrlPoints; ii++){

			float2 dir; //tangent
			if (ii == numCtrlPoints - 1)
				dir = normalize(ctrlPoints[numCtrlPoints - 1] - ctrlPoints[numCtrlPoints - 2]);
			else if (ii == 0)
				dir = normalize(ctrlPoints[1] - ctrlPoints[0]);
			else
				dir = normalize((ctrlPoints[ii + 1] - ctrlPoints[ii - 1]) / 2);
			float2 normal = make_float2(-dir.y, dir.x);

			sidePointsPos.push_back(center + ctrlPoints[ii] + normal * width);
			sidePointsNeg.push_back(center + ctrlPoints[ii] - normal * width);
		}


		std::vector<float2> posBezierPoints = BezierSmaple(sidePointsPos);
		std::vector<float2> negBezierPoints = BezierSmaple(sidePointsNeg);
		std::reverse(negBezierPoints.begin(), negBezierPoints.end());
		ret.insert(ret.begin(), posBezierPoints.begin(), posBezierPoints.end());
		ret.insert(ret.end(), negBezierPoints.begin(), negBezierPoints.end());
	}
	return ret;
}


std::vector<float2> CurveBLens::GetExtraLensRendering(){
	std::vector<float2> ret;
	if (isConstructing){
		for (int ii = 0; ii < numCtrlPoints; ii++) {
			ret.push_back(make_float2(ctrlPointsAbs[ii].x, ctrlPointsAbs[ii].y));
		}
	}
	else{
		float2 center = make_float2(x, y);
		int n = ctrlPoints.size();
		for (int ii = 0; ii < n; ii++) {
			ret.push_back(ctrlPoints[ii] + center);
		}
	}
	return ret;
}

std::vector<float2> CurveBLens::GetExtraLensRendering2(){
	std::vector<float2> ret;
	if (isConstructing){
		for (int ii = 0; ii < numCtrlPoints; ii++) {
			ret.push_back(make_float2(ctrlPointsAbs[ii].x, ctrlPointsAbs[ii].y));
		}
	}
	else{
		float2 center = make_float2(x, y);
		int n = BezierPoints.size();
		for (int ii = 0; ii < n; ii++) {
			ret.push_back(BezierPoints[ii] + center);
		}
	}
	return ret;
}

void CurveBLens::RefineLensCenter()
{
	//currently process the positive one first
	int n = rationalCtrlPoints.size();//may contain more number than numCtrlPoints
	if (n < 2)
		return;
	if (posOffsetBezierPoints.size() != n)
	{
		cout << "error! " << endl;
		return;
	}

	bool justSplitted = false;

	vector<float2> newRationalCtrlPoints;
	bool convergedPos = true;
	float thr = 0.05;
	for (int i = 0; i < n; i++){
		//may need to be improved !!!
		if (abs(length(BezierPoints[i] - posOffsetBezierPoints[i]) - width)<width*thr){
			newRationalCtrlPoints.push_back(rationalCtrlPoints[i]);
			justSplitted = true;
		}
		else{/*
			if (i>0 && !justSplitted){
				newRationalCtrlPoints.push_back((rationalCtrlPoints[i] + rationalCtrlPoints[i - 1]) / 2);
			}
			//newRationalCtrlPoints.push_back(rationalCtrlPoints[i]);
			if (i < n - 1){
				newRationalCtrlPoints.push_back((rationalCtrlPoints[i] + rationalCtrlPoints[i + 1]) / 2);
			}*/
			newRationalCtrlPoints.push_back(((rationalCtrlPoints[i] + rationalCtrlPoints[i - 1]) / 2 + (rationalCtrlPoints[i] + rationalCtrlPoints[i + 1]) / 2) / 2);
			convergedPos = false;
			justSplitted = false;
		}
	}
	if (!convergedPos){
		cout << "Did once! from " << n << " to " << newRationalCtrlPoints.size()<< endl;
		rationalCtrlPoints.clear();
		rationalCtrlPoints = newRationalCtrlPoints;
		posOffsetCtrlPoints.clear();
		//compute posOffsetCtrlPoints
		int n = rationalCtrlPoints.size();
		float2 normal;
		float2 dir;

		dir = normalize(rationalCtrlPoints[1] - rationalCtrlPoints[0]);
		normal = make_float2(-dir.y, dir.x);
		posOffsetCtrlPoints.push_back(rationalCtrlPoints[0] + normal*width);

		for (int i = 1; i < n - 1; i++){
			dir = normalize(rationalCtrlPoints[i] - rationalCtrlPoints[i - 1]);
			normal = make_float2(-dir.y, dir.x);
			float2 dir2 = normalize(rationalCtrlPoints[i + 1] - rationalCtrlPoints[i]);
			float2 normal2 = make_float2(-dir.y, dir.x);
			float2 posCtrlPoint;
			intersectPoint(rationalCtrlPoints[i - 1] + normal*width, rationalCtrlPoints[i] + normal*width, rationalCtrlPoints[i] + normal2*width, rationalCtrlPoints[i + 1] + normal2*width, posCtrlPoint);
			posOffsetCtrlPoints.push_back(posCtrlPoint);
		}

		dir = normalize(rationalCtrlPoints[n - 1] - rationalCtrlPoints[n - 2]);
		normal = make_float2(-dir.y, dir.x);
		posOffsetCtrlPoints.push_back(rationalCtrlPoints[n - 1] + normal*width);

		posOffsetBezierPoints = BezierSmaple(posOffsetCtrlPoints);
	}
	else{
		cout << "converged!" << endl;
	}
}


void CurveBLens::RefineLensBoundary()
{
	posOffsetCtrlPoints = removeSelfIntersection(posOffsetCtrlPoints, true);
	negOffsetCtrlPoints = removeSelfIntersection(negOffsetCtrlPoints, true);

	posOffsetBezierPoints = BezierSmaple(posOffsetCtrlPoints);
	negOffsetBezierPoints = BezierSmaple(negOffsetCtrlPoints);
}

vector<float2> CurveBLens::removeSelfIntersection(vector<float2> p, bool isDuplicating)
{
	int n = p.size();
	bool *skipped = new bool[n];
	float2 *itsPoints = new float2[n];

	for (int i = 0; i < n; i++){
		skipped[i] = false;
	}
	for (int i = 2; i < n-1; i++){
		bool notFoundIntersect = true;
		for (int j = 0; j < i - 1 && notFoundIntersect; j++){
			if (!skipped[j]){
				/// !!! NOTE: only valid for one round of refine. since the improved ctrl points has multiplicates of the intersected points, it will cause the if condition be naturally satisfied !!!
				if (intersect(p[j], p[j + 1], p[i], p[i + 1])){
					float2 intersectP;
					intersectPoint(p[j], p[j + 1], p[i], p[i + 1], intersectP);
					for (int k = j; k <= i; k++){
						skipped[k] = true;
						itsPoints[k] = intersectP;
						notFoundIntersect = false;
					}
				}
			}
		}
	}
	vector<float2> newp;
	newp.push_back(p[0]);
	if (isDuplicating){
		for (int i = 0; i < n-1; i++){
			if (!skipped[i]){
				newp.push_back(p[i+1]);
			}
			else{
				newp.push_back(itsPoints[i]);
			}
		}
	}
	else{
		bool lastSegSkipped = false;
		for (int i = 0; i < n-1; i++){
			if (!skipped[i]){
				newp.push_back(p[i+1]);
				lastSegSkipped = false;
			}
			else{
				if (!lastSegSkipped){
					newp.push_back(itsPoints[i]);
					lastSegSkipped = true;
				}
			}
		}
	}
	return newp;
}
