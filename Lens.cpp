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

		float2 center = make_float2(x, y);
		vector<float2> BezierSmapleAbs = BezierSmaple(ctrlPointsAbs);
		BezierCtrlPoints.resize(numCtrlPoints);
		for (int ii = 0; ii < numCtrlPoints; ii++) {
			BezierCtrlPoints[ii] = BezierSmapleAbs[ii] - center;
		}
		isConstructing = false;

		/*
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

		*/
	}
}



std::vector<float2> CurveBLens::GetContour(){
	std::vector<float2> ret;
	return ret;
	if (!isConstructing && numCtrlPoints >= 3) {
		float2 center = make_float2(x, y);

		ret.resize(2 * numCtrlPoints);
		for (int ii = 0; ii < numCtrlPoints; ii++){

			float2 dir; //tangent
			if (ii == numCtrlPoints - 1)
				dir = normalize(ctrlPoints[numCtrlPoints - 1] - ctrlPoints[numCtrlPoints - 2]);
			else if (ii == 0)
				dir = normalize(ctrlPoints[1] - ctrlPoints[0]);
			else
				dir = normalize((ctrlPoints[ii + 1] - ctrlPoints[ii - 1]) / 2);
			float2 normal = make_float2(-dir.y, dir.x);

			ret[ii] = center + ctrlPoints[ii] + normal * width;
			ret[2 * numCtrlPoints - 1 - ii] = center + ctrlPoints[ii] - normal * width;
		}


	}


	/*		if (!isConstructing && numCtrlPoints >= 3) {
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
	*/
	return ret;
}


std::vector<float2> CurveBLens::GetOuterContour() {
	std::vector<float2> ret;

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

			sidePointsPos.push_back(center + ctrlPoints[ii] + normal * width *1.1);
			sidePointsNeg.push_back(center + ctrlPoints[ii] - normal * width *1.1);
		}


		std::vector<float2> posBezierPoints = BezierSmaple(sidePointsPos);
		std::vector<float2> negBezierPoints = BezierSmaple(sidePointsNeg);
		std::reverse(negBezierPoints.begin(), negBezierPoints.end());
		ret.insert(ret.begin(), posBezierPoints.begin(), posBezierPoints.end());
		ret.insert(ret.end(), negBezierPoints.begin(), negBezierPoints.end());
	}
	return ret;
}

std::vector<float2> CurveBLens::GetOuterContourold() {
	std::vector<float2> ret;
	return ret;
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

std::vector<float2> CurveBLens::GetExtraLensRendering(){
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

std::vector<float2> CurveBLens::GetExtraLensRendering2(){
	std::vector<float2> ret;
	if (isConstructing){
		for (int ii = 0; ii < numCtrlPoints; ii++) {
			ret.push_back(make_float2(ctrlPointsAbs[ii].x, ctrlPointsAbs[ii].y));
		}
	}
	else{
		//float2 center = make_float2(x, y);
		//for (int ii = 0; ii < numCtrlPoints; ii++) {
		//	ctrlPointsAbs[ii] = ctrlPoints[ii] + center;
		//}
		//ret = BezierSmaple(ctrlPointsAbs);

		float2 center = make_float2(x, y);
		int n = BezierCtrlPoints.size();
		for (int ii = 0; ii < n; ii++) {
			ret.push_back(BezierCtrlPoints[ii] + center);
		}
	}
	return ret;
}

void CurveBLens::RefineLensCenter()
{
	BezierPosOffset;
	std::vector<float2> BezierNegPoints


	int n = BezierCtrlPoints.size();//may contain more number than numCtrlPoints
	if (n>1){
		vector<float2> newBezierPosOffset;
		float2 normal;

		float2 dir; //tangent

		dir = normalize(ctrlPoints[1] - ctrlPoints[0]);
		normal = make_float2(-dir.y, dir.x);

		newBezierPosOffset.push_back(BezierCtrlPoints[0] + normal*width);

		for (int i = 1; i < n-1; i++){
			dir = normalize(ctrlPoints[i] - ctrlPoints[i - 1]);
			normal = make_float2(-dir.y, dir.x);
			float2 dir2 = normalize(ctrlPoints[i + 1] - ctrlPoints[i]);
			float2 normal2 = make_float2(-dir.y, dir.x);

		}


		cout << "did once" << endl;
	}
}