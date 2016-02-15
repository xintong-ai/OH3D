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
		
		//remove end shaking
		float endRegionThr = 0.15;
		int end1ind = ceil(numCtrlPoints*endRegionThr);
		if (end1ind > 0 && end1ind < numCtrlPoints){
			float dis = length(ctrlPointsAbs[end1ind] - ctrlPointsAbs[0]);
			float2 dir = normalize(ctrlPointsAbs[end1ind] - ctrlPointsAbs[0]);
			for (int i = 1; i < end1ind; i++){
				ctrlPointsAbs[i] = ctrlPointsAbs[0] + dis*i / end1ind*dir;
			}
		}
		int end2ind = numCtrlPoints*(1-endRegionThr)-1;
		if (end2ind >= 0 && end2ind < numCtrlPoints - 1){
			float dis = length(ctrlPointsAbs[numCtrlPoints - 1] - ctrlPointsAbs[end2ind]);
			float2 dir = normalize(ctrlPointsAbs[numCtrlPoints - 1] - ctrlPointsAbs[end2ind]);
			for (int i = end2ind + 1; i < numCtrlPoints - 1; i++){
				ctrlPointsAbs[i] = ctrlPointsAbs[end2ind] + dis*(i - end2ind) / (numCtrlPoints - 1 - end1ind)*dir;
			}
		}
		//cout << numCtrlPoints << " " << end1ind << " " << end2ind << endl;


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

		//compute the BezierPoints used to draw the curve
		BezierPoints = BezierSmaple(ctrlPoints);
		//BezierPoints = BSplineSubdivide(ctrlPoints,4,0.5);

		//compute the boundary

		//use another array to store the control points, since it might be changed later to refine the offset
		subCtrlPointsPos = ctrlPoints;
		subCtrlPointsNeg = ctrlPoints;
		refinedRoundPos = 0;
		refinedRoundNeg = 0;

		offsetControlPointsPos();
		offsetControlPointsNeg();
		computeBoundaryPos();
		computeBoundaryNeg();
		

		while (adjustOffset() && refinedRoundPos < refineIterationLimit && refinedRoundNeg < refineIterationLimit);

		posOffsetCtrlPoints = removeSelfIntersection(posOffsetCtrlPoints, true);
		negOffsetCtrlPoints = removeSelfIntersection(negOffsetCtrlPoints, true);
		computeBoundaryPos();
		computeBoundaryNeg();

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

void CurveBLens::offsetControlPointsPos()
{
	float2 normal;
	float2 dir;

	int np = subCtrlPointsPos.size();
	posOffsetCtrlPoints.clear(); //!!! maybe can improve performance by using vector<>::resize??

	dir = normalize(subCtrlPointsPos[1] - subCtrlPointsPos[0]);
	normal = make_float2(-dir.y, dir.x);
	posOffsetCtrlPoints.push_back(subCtrlPointsPos[0] + normal*width);

	for (int i = 1; i < np - 1; i++){
		dir = normalize(subCtrlPointsPos[i] - subCtrlPointsPos[i - 1]);
		normal = make_float2(-dir.y, dir.x);
		float2 dir2 = normalize(subCtrlPointsPos[i + 1] - subCtrlPointsPos[i]);
		float2 normal2 = make_float2(-dir.y, dir.x);
		float2 posCtrlPoint;
		intersectPoint(subCtrlPointsPos[i - 1] + normal*width, subCtrlPointsPos[i] + normal*width, subCtrlPointsPos[i] + normal2*width, subCtrlPointsPos[i + 1] + normal2*width, posCtrlPoint);
		posOffsetCtrlPoints.push_back(posCtrlPoint);
	}

	dir = normalize(subCtrlPointsPos[np - 1] - subCtrlPointsPos[np - 2]);
	normal = make_float2(-dir.y, dir.x);
	posOffsetCtrlPoints.push_back(subCtrlPointsPos[np - 1] + normal*width);
}

void CurveBLens::offsetControlPointsNeg()
{
	float2 normal;
	float2 dir;
		
	int nn = subCtrlPointsNeg.size();
	negOffsetCtrlPoints.clear();

	dir = normalize(subCtrlPointsNeg[1] - subCtrlPointsNeg[0]);
	normal = make_float2(-dir.y, dir.x);
	negOffsetCtrlPoints.push_back(subCtrlPointsNeg[0] - normal*width);

	for (int i = 1; i < nn - 1; i++){
		dir = normalize(subCtrlPointsNeg[i] - subCtrlPointsNeg[i - 1]);
		normal = make_float2(-dir.y, dir.x);
		float2 dir2 = normalize(subCtrlPointsNeg[i + 1] - subCtrlPointsNeg[i]);
		float2 normal2 = make_float2(-dir.y, dir.x);

		float2 negCtrlPoint;
		intersectPoint(subCtrlPointsNeg[i - 1] - normal*width, subCtrlPointsNeg[i] - normal*width, subCtrlPointsNeg[i] - normal2*width, subCtrlPointsNeg[i + 1] - normal2*width, negCtrlPoint);
		negOffsetCtrlPoints.push_back(negCtrlPoint);
	}

	dir = normalize(subCtrlPointsNeg[nn - 1] - subCtrlPointsNeg[nn - 2]);
	normal = make_float2(-dir.y, dir.x);
	negOffsetCtrlPoints.push_back(subCtrlPointsNeg[nn - 1] - normal*width);

}


void CurveBLens::computeBoundaryPos()
{
	int np = posOffsetCtrlPoints.size();
	int numberPosBezierPart = pow(2, refinedRoundPos);
	posOffsetBezierPoints.clear();
	for (int i = 0; i < numberPosBezierPart; i++){
		vector<float2> curPart(
			posOffsetCtrlPoints.begin() + (np - 1) / numberPosBezierPart*i,
			posOffsetCtrlPoints.begin() + (np - 1) / numberPosBezierPart*(i + 1) + 1);
		vector<float2> curBezierPart = BezierSmaple(curPart);
		if (i == 0)
			posOffsetBezierPoints.insert(posOffsetBezierPoints.end(), curBezierPart.begin(), curBezierPart.end());
		else
			posOffsetBezierPoints.insert(posOffsetBezierPoints.end(), curBezierPart.begin() + 1, curBezierPart.end());
	}

}

void CurveBLens::computeBoundaryNeg()
{
	int nn = negOffsetCtrlPoints.size();
	int numberNegBezierPart = pow(2, refinedRoundNeg);
	negOffsetBezierPoints.clear();
	for (int i = 0; i < numberNegBezierPart; i++){
		vector<float2> curPart(
			negOffsetCtrlPoints.begin() + (nn - 1) / numberNegBezierPart*i,
			negOffsetCtrlPoints.begin() + (nn - 1) / numberNegBezierPart*(i + 1) + 1);
		vector<float2> curBezierPart = BezierSmaple(curPart);
		if (i == 0)
			negOffsetBezierPoints.insert(negOffsetBezierPoints.end(), curBezierPart.begin(), curBezierPart.end());
		else
			negOffsetBezierPoints.insert(negOffsetBezierPoints.end(), curBezierPart.begin() + 1, curBezierPart.end());
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

bool CurveBLens::adjustOffset()
{
	if (subCtrlPointsPos.size() < 2 || subCtrlPointsNeg.size() < 2)
		return false;
	bool convergedPos = true;
	bool convergedNeg = true;
	float offsetDisThr = 0.05;

	//process the positive one first
	{
		int oriN = BezierPoints.size();
		int posOBPsize = posOffsetBezierPoints.size();
		for (int i = 0; i < oriN; i++){
			//may need to be improved !!!
			if (abs(length(BezierPoints[i] - posOffsetBezierPoints[round(i *1.0 / (oriN - 1)*(posOBPsize - 1))]) - width) < width*offsetDisThr){
				//newRationalCtrlPoints.push_back(subCtrlPointsPos[i]);
			}
			else{
				convergedPos = false;
			}
		}
		if (!convergedPos){
			int m = subCtrlPointsPos.size();
			int numberPosBezierPart = pow(2, refinedRoundPos);

			vector<float2> newRCP;
			for (int i = 0; i < numberPosBezierPart; i++){
				vector<float2> curPart(
					subCtrlPointsPos.begin() + (m - 1) / numberPosBezierPart*i,
					subCtrlPointsPos.begin() + (m - 1) / numberPosBezierPart*(i + 1) + 1);
				vector<float2> refinedCurPart = BezierSubdivide(curPart, 1, 0.5);
				if (i == 0)
					newRCP.insert(newRCP.end(), refinedCurPart.begin(), refinedCurPart.end());
				else
					newRCP.insert(newRCP.end(), refinedCurPart.begin() + 1, refinedCurPart.end());

			}
			subCtrlPointsPos = newRCP;
			refinedRoundPos++;

			offsetControlPointsPos();
			computeBoundaryPos();

			cout << "refine for pos boundary once! count boundary points: " << subCtrlPointsPos.size() << endl;
		}
		else{
			cout << "pos converged!" << endl;
		}
	}


	//process the negative one
	{
		int oriN = BezierPoints.size();
		int negOBPsize = negOffsetBezierPoints.size();
		for (int i = 0; i < oriN; i++){
			//may need to be improved !!!
			if (abs(length(BezierPoints[i] - negOffsetBezierPoints[round(i *1.0 / (oriN - 1)*(negOBPsize - 1))]) - width) < width*offsetDisThr){
				//newRationalCtrlPoints.push_back(subCtrlPointsPos[i]);
			}
			else{
				convergedNeg = false;
			}
		}
		if (!convergedNeg){
			int m = subCtrlPointsNeg.size();
			int numberNegBezierPart = pow(2, refinedRoundNeg);

			vector<float2> newRCP;
			for (int i = 0; i < numberNegBezierPart; i++){
				vector<float2> curPart(
					subCtrlPointsNeg.begin() + (m - 1) / numberNegBezierPart*i,
					subCtrlPointsNeg.begin() + (m - 1) / numberNegBezierPart*(i + 1) + 1);
				vector<float2> refinedCurPart = BezierSubdivide(curPart, 1, 0.5);
				if (i == 0)
					newRCP.insert(newRCP.end(), refinedCurPart.begin(), refinedCurPart.end());
				else
					newRCP.insert(newRCP.end(), refinedCurPart.begin() + 1, refinedCurPart.end());

			}
			subCtrlPointsNeg = newRCP;
			refinedRoundNeg++;

			offsetControlPointsNeg();
			computeBoundaryNeg();

			cout << "refine for neg boundary once! count boundary points:" << subCtrlPointsNeg.size() << endl;
		}
		else{
			cout << "neg converged!" << endl;
		}
	}
	if (!convergedPos || !convergedNeg)
		return true;
	return false;
}


void CurveBLens::RefineLensBoundary()
{
	posOffsetCtrlPoints = removeSelfIntersection(posOffsetCtrlPoints, true);
	negOffsetCtrlPoints = removeSelfIntersection(negOffsetCtrlPoints, true);

	computeBoundaryPos();
	computeBoundaryNeg();
}

vector<float2> CurveBLens::removeSelfIntersection(vector<float2> p, bool isDuplicating)
{
	int n = p.size();
	bool *skipped = new bool[n];
	float2 *itsPoints = new float2[n];

	bool notFoundOverall = true;

	for (int i = 0; i < n; i++){
		skipped[i] = false;
	}
	/// !!! NOTE: this setting, and the intersectPoint() function treats not intersected when two segments share a same end point. cases may fail if the curve's self intersection happens at a control point !!!
	float lengthThr = 0.00001;
	for (int i = 2; i < n - 1; i++){
		if (length(p[i] - p[i + 1])>lengthThr){
			bool notFoundIntersect = true;
			for (int j = 0; j < i - 1 && notFoundIntersect; j++){
				if (!skipped[j] && length(p[j] - p[j + 1])>lengthThr){
					if (intersect(p[j], p[j + 1], p[i], p[i + 1])){
						float2 intersectP;
						intersectPoint(p[j], p[j + 1], p[i], p[i + 1], intersectP);
						for (int k = j; k <= i; k++){
							skipped[k] = true;
							itsPoints[k] = intersectP;
							notFoundIntersect = false;
							notFoundOverall = false;
						}
					}
				}
			}
		}
	}

	if (notFoundOverall){
		cout << "no self intersection" << endl;
		return p;
	}
	else
	{
		cout << "did self intersection removal once!" << endl;
		vector<float2> newp;
		newp.push_back(p[0]);
		if (isDuplicating){
			for (int i = 0; i < n - 2; i++){
				if (!skipped[i]){
					newp.push_back(p[i + 1]);
				}
				else{
					newp.push_back(itsPoints[i]);
				}
			}
		}
		else{
			bool lastSegSkipped = false;
			for (int i = 0; i < n - 2; i++){
				if (!skipped[i]){
					newp.push_back(p[i + 1]);
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
		newp.push_back(p[n-1]);
		return newp;
	}
}


vector<float2> CurveBLens::BezierOneSubdivide(vector<float2> p, vector<float2> poly1, vector<float2> poly2, float u)
{
	vector<float2> res;
	int n = p.size();
	if (n == 1)
	{
		res.insert(res.begin(), poly1.begin(), poly1.end());
		res.insert(res.end(), p[0]);
		res.insert(res.end(), poly2.begin(), poly2.end());
	}
	else if (p.size() > 1){
		poly1.push_back(p[0]);
		vector<float2> newpoly2;
		newpoly2.push_back(p[n - 1]);
		newpoly2.insert(newpoly2.end(), poly2.begin(), poly2.end());

		vector<float2> newp;
		for (int i = 0; i < n - 1; i++) {
			newp.push_back(p[i] + u*(p[i + 1] - p[i]));
		}
		res = BezierOneSubdivide(newp, poly1, newpoly2, u);

	}
	return res;

}

vector<float2> CurveBLens::BezierSubdivide(vector<float2> p, int m, float u)
{
	vector<float2> res;
	if (m == 1) {
		vector<float2> poly1(0), poly2(0);
		res = BezierOneSubdivide(p, poly1, poly2, u);
	}
	else {
		vector<float2> poly1(0), poly2(0);
		vector<float2> doubledp = BezierOneSubdivide(p, poly1, poly2, u);

		int n = (doubledp.size() + 1) / 2;
		vector<float2> newpLeft(doubledp.begin(), doubledp.begin() + n);
		vector<float2> newpRight(doubledp.begin() + n - 1, doubledp.end());

		vector<float2> respLeft = BezierSubdivide(newpLeft, m - 1, u);
		vector<float2> respRight = BezierSubdivide(newpRight, m - 1, u);

		res.insert(res.begin(), respLeft.begin(), respLeft.end());
		res.insert(res.end(), respRight.begin() + 1, respRight.end());
	}
	return res;
}

vector<float2> CurveBLens::BezierSmaple(vector<float2> p)
{
	vector<float2> res;
	if (p.size() >= 2){
#define bezierSampleAccuracyRate 1 
		int n = p.size() - 1;

		double *combinationValue = new double[n + 1];
		for (int i = 0; i <= n / 2; i++){
			double cc = 1; //compute n!/i!/(n-i)! = ((n-i+1)*...*n)/(1*2*...*i)
			for (int j = i; j >= 1; j--){
				cc = cc*(n + j - i) / j;
			}
			combinationValue[i] = cc;
		}
		for (int i = n / 2 + 1; i <= n; i++){
			combinationValue[i] = combinationValue[n - i];
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

vector<float2> CurveBLens::BSplineOneSubdivide(vector<float2> p, int m, float u)
{
	vector<float2> res;
	if (m == 1){
		res.push_back(p[0] * 0.75 + p[1] * 0.25);
		res.push_back(p[0] * 0.25 + p[1] * 0.75);
		res.push_back(p[1] * 0.75 + p[2] * 0.25);
		res.push_back(p[1] * 0.25 + p[2] * 0.75);
	}
	else
	{
		vector<float2> p1(3), p2(3);
		p1[0] = 0.75*p[0] + 0.25*p[1];
		p1[1] = 0.25*p[0] + 0.75*p[1];
		p1[2] = 0.75*p[1] + 0.25*p[2];
		p2[0] = p1[1];
		p2[1] = p1[2];
		p2[2] = 0.25*p[1] + 0.75*p[2];

		vector<float2> res1 = BSplineOneSubdivide(p1, m - 1, u);
		vector<float2> res2 = BSplineOneSubdivide(p2, m - 1, u);

		res = res1;
		res.pop_back();
		res.pop_back();
		res.insert(res.end(), res2.begin(), res2.end());
	}
	return res;
}

vector<float2> CurveBLens::BSplineSubdivide(vector<float2> p, int m, float u)
{
	int D = 3;
	vector<float2> res;
	for (int i = 0; i < p.size() - 2; i++){
		vector<float2> pcur(3);
		pcur[0] = p[i];
		pcur[1] = p[i + 1];
		pcur[2] = p[i + 2];
		vector<float2> rescur = BSplineOneSubdivide(pcur, m, u);
		if (i == 0)
			res = rescur;
		else{
			res.pop_back();
			res.pop_back();
			res.insert(res.end(), rescur.begin(), rescur.end());
		}
	}
	return res;
}
