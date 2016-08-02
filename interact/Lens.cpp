#include "Lens.h"
#include "TransformFunc.h"
#include <iostream>
#include <algorithm>

float Lens::GetClipDepth(float* mv, float* pj)
{
	return Object2Clip(GetCenter(), mv, pj).z;
}

float2 Lens::GetCenterScreenPos(float* mv, float* pj, int winW, int winH)
{
	return Object2Screen(GetCenter(), mv, pj, winW, winH);
}

void Lens::UpdateCenterByScreenPos(int sx, int sy, float* mv, float* pj, int winW, int winH)
{
	matrix4x4 invModelview, invProjection;
	invertMatrix(mv, &invModelview.v[0].x);
	invertMatrix(pj, &invProjection.v[0].x);
	float4 cenClip = Object2Clip(GetCenter(), mv, pj);
	float2 newClipXY = Screen2Clip(make_float2(sx, sy), winW, winH);
	float4 newClip = make_float4(newClipXY.x, newClipXY.y, cenClip.z, cenClip.w);
	float4 newObject = Clip2ObjectGlobal(newClip, &invModelview.v[0].x, &invProjection.v[0].x);
	SetCenter(make_float3(newObject));
}

float3 Lens::Compute3DPosByScreenPos(int sx, int sy, float* mv, float* pj, int winW, int winH)
{
	matrix4x4 invModelview, invProjection;
	invertMatrix(mv, &invModelview.v[0].x);
	invertMatrix(pj, &invProjection.v[0].x);
	float4 cenClip = Object2Clip(GetCenter(), mv, pj);
	float2 newClipXY = Screen2Clip(make_float2(sx, sy), winW, winH);
	float4 newClip = make_float4(newClipXY.x, newClipXY.y, cenClip.z, cenClip.w);
	float4 newObject = Clip2ObjectGlobal(newClip, &invModelview.v[0].x, &invProjection.v[0].x);
	//SetCenter(make_float3(newObject));
	return make_float3(newObject);
}


float4 Lens::GetCenter() { return make_float4(c.x, c.y, c.z, 1.0f); }

void Lens::SetClipDepth(float d, float* mv, float* pj)
{
	matrix4x4 invModelview, invProjection;
	invertMatrix(mv, &invModelview.v[0].x);
	invertMatrix(pj, &invProjection.v[0].x);
	float4 cenClip = Object2Clip(GetCenter(), mv, pj);
	//std::cout << "cenClip.z:" << cenClip.z << std::endl;
	cenClip.z = d;
//	float4 cenShiftClip = cenClip + make_float4(0, 0, -0.01, 0);
	float4 cenShiftObj = Clip2ObjectGlobal(cenClip, &invModelview.v[0].x, &invProjection.v[0].x);
	//float4 dir4 = cenShiftObj - GetCenter();// make_float3(dir_object.x, dir_object.y, dir_object.z);
	//float3 dir3 = make_float3(dir4.x, dir4.y, dir4.z);
	//dir3 = dir3 * (1.0f / length(dir3)) * v * (-0.05);
	SetCenter(make_float3(cenShiftObj));
}

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




std::vector<std::vector<float3>> CircleLens::Get3DContour(float3 eyeWorld, bool isScreenDeformingLens)
{
	std::vector<std::vector<float3>> contour3D;
	//return contour3D;

	if (isScreenDeformingLens){ //draw screen-space deformed circle lens
		float3 v = normalize(eyeWorld - c);
		float3 tempdir;
		tempdir = make_float3(0, 0, 1);
		if (dot(tempdir, v)>0.9)
			tempdir = make_float3(0, 1, 0);
		float3 xdir = cross(tempdir, v);
		float3 ydir = cross(xdir, v);

		std::vector<float3> innerContour;
		std::vector<float3> outerContour;
		std::vector<float3> bottomContour;
		std::vector<float3> connection;

		float rr = 2; //need to transfer screen radius to object radias
		float d1 = 2, d2 = 2;
		const int num_segments = 32;
		for (int ii = 0; ii < num_segments; ii++)
		{
			float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);//get the current angle 

			float ax = rr * cosf(theta);//calculate the x component 
			float ay = rr * sinf(theta);//calculate the y component 
			float ax2 = rr / focusRatio * cosf(theta);//calculate the x component 
			float ay2 = rr / focusRatio * sinf(theta);//calculate the y component 

			float3 ip = c + ax*xdir + ay*ydir;
			float3 op = c + ax2*xdir + ay2*ydir + v*d1;
			float3 bp = c + ax*xdir + ay*ydir - v*d2;

			innerContour.push_back(ip);
			outerContour.push_back(op);
			bottomContour.push_back(bp);

			if (ii % 4 == 0){
				connection.push_back(ip);
				connection.push_back(op);

				connection.push_back(ip);
				connection.push_back(bp);
			}
		}

		contour3D.push_back(innerContour);
		contour3D.push_back(outerContour);
		contour3D.push_back(bottomContour);
		contour3D.push_back(connection);
	}
	else{
		float3 v = normalize(eyeWorld - c);
		float3 tempdir;
		tempdir = make_float3(0, 0, 1);
		if (dot(tempdir, v)>0.9)
			tempdir = make_float3(0, 1, 0);
		float3 xdir = cross(tempdir, v);
		float3 ydir = cross(xdir, v);

		std::vector<float3> innerContour;
		std::vector<float3> outerContour;
		std::vector<float3> topInnerContour;
		std::vector<float3> topOuterContour;
		std::vector<float3> innerConnection;
		std::vector<float3> outerConnection;

		float rr = objectRadius;
		float d1 = 10;// , d2 = 20;
		const int num_segments = 32;
		for (int ii = 0; ii < num_segments; ii++)
		{
			float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);//get the current angle 

			float ax = rr * cosf(theta);//calculate the x component 
			float ay = rr * sinf(theta);//calculate the y component 
			float ax2 = rr / focusRatio * cosf(theta);//calculate the x component 
			float ay2 = rr / focusRatio * sinf(theta);//calculate the y component 

			float3 ip = c + ax*xdir + ay*ydir;
			float3 op = c + ax2*xdir + ay2*ydir;
			float3 tip = c + ax*xdir + ay*ydir + v*d1;
			float3 top = c + ax2*xdir + ay2*ydir + v*d1;

			innerContour.push_back(ip);
			outerContour.push_back(op);
			topInnerContour.push_back(tip);
			topOuterContour.push_back(top);

			if (ii % 4 == 0){
				innerConnection.push_back(ip);
				innerConnection.push_back(tip);
			}
			else if (ii % 4 == 2){
				outerConnection.push_back(op);
				outerConnection.push_back(top);
			}
		}

		contour3D.push_back(innerContour);
		contour3D.push_back(topInnerContour);
		contour3D.push_back(outerContour);
		contour3D.push_back(topOuterContour);
		contour3D.push_back(innerConnection);
		contour3D.push_back(outerConnection);
	}
	
	return contour3D;
}

bool CircleLens::PointInsideObjectLens(int _x, int _y, float* mv, float* pj, int winW, int winH) {
	float3 clickPoint = Compute3DPosByScreenPos(_x, _y, mv, pj, winW, winH);
	return length(c - clickPoint) < objectRadius;
}

bool CircleLens::PointOnObjectInnerBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH)
{
	float3 clickPoint = Compute3DPosByScreenPos(_x, _y, mv, pj, winW, winH);
	float eps_dis = objectRadius*0.1;
	float dis = length(c - clickPoint);
	return std::abs(dis - objectRadius) < eps_dis;
}

bool CircleLens::PointOnObjectOuterBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH)
{
	float3 clickPoint = Compute3DPosByScreenPos(_x, _y, mv, pj, winW, winH);
	float eps_dis = objectRadius*0.1 / focusRatio;
	float dis = length(c - clickPoint);
	return std::abs(dis - objectRadius/focusRatio) < eps_dis;
}

void CircleLens::ChangeObjectLensSize(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)
{
	float3 clickPoint = Compute3DPosByScreenPos(_x, _y, mv, pj, winW, winH);
	objectRadius = length(c - clickPoint);
}

void CircleLens::ChangeObjectFocusRatio(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)
{
	float3 clickPoint = Compute3DPosByScreenPos(_x, _y, mv, pj, winW, winH);
	focusRatio = objectRadius / length(c - clickPoint);
}





bool LineBLens::PointInsideLens(int _x, int _y, float* mv, float* pj, int winW, int winH) {


	//dot product of (_x-x, _y-y) and direction
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);
	float2 toPoint = make_float2(_x - center.x, _y - center.y);
	float disMajor = toPoint.x*direction.x + toPoint.y*direction.y;

	if (std::abs(disMajor) < lSemiMajorAxis) {
		float2 minorDirection = make_float2(-direction.y, direction.x);
		//dot product of (_x-x, _y-y) and minorDirection
		float disMinor = (_x - center.x)*minorDirection.x + (_y - center.y)*minorDirection.y;
		if (std::abs(disMinor) < lSemiMinorAxis)
			return true;
	}
	return false;
}



std::vector<float2> LineBLens::GetContour(float* mv, float* pj, int winW, int winH)
{
	std::vector<float2> ret;
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);
	float2 ctrlPoint1 = center - direction*lSemiMajorAxis;
	float2 ctrlPoint2 = center + direction*lSemiMajorAxis;

	float2 minorDirection = make_float2(-direction.y, direction.x);


	ret.push_back(ctrlPoint1 - minorDirection*lSemiMinorAxis);
	ret.push_back(ctrlPoint2 - minorDirection*lSemiMinorAxis);
	ret.push_back(ctrlPoint2 + minorDirection*lSemiMinorAxis);
	ret.push_back(ctrlPoint1 + minorDirection*lSemiMinorAxis);
	return ret;
}

std::vector<float2> LineBLens::GetOuterContour(float* mv, float* pj, int winW, int winH)
{
	std::vector<float2> ret;
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);
	float2 ctrlPoint1 = center - direction*lSemiMajorAxis;
	float2 ctrlPoint2 = center + direction*lSemiMajorAxis;

	float2 minorDirection = make_float2(-direction.y, direction.x);

	ret.push_back(ctrlPoint1 - minorDirection*lSemiMinorAxis / focusRatio);
	ret.push_back(ctrlPoint2 - minorDirection*lSemiMinorAxis / focusRatio);
	ret.push_back(ctrlPoint2 + minorDirection*lSemiMinorAxis / focusRatio);
	ret.push_back(ctrlPoint1 + minorDirection*lSemiMinorAxis / focusRatio);

	return ret;
}
void LineBLens::UpdateLineBLensInfo()
{
	lineBLensInfo.lSemiMajorAxis = lSemiMajorAxis;
	lineBLensInfo.lSemiMinorAxis = lSemiMinorAxis;
	lineBLensInfo.direction = direction;
	lineBLensInfo.focusRatio = focusRatio;
}

void LineBLens::UpdateInfo(float* mv, float* pj, int winW, int winH)
{
	UpdateCenterByScreenPos((ctrlPoint1Abs.x + ctrlPoint2Abs.x) / 2.0, (ctrlPoint1Abs.y + ctrlPoint2Abs.y) / 2.0, mv, pj, winW, winH);

	direction = ctrlPoint2Abs - ctrlPoint1Abs;

	lSemiMajorAxis = length(direction) / 2;
	float ratio = 3.0f;
	lSemiMinorAxis = lSemiMajorAxis / ratio;

	if (lSemiMajorAxis < 0.000001)
		direction = make_float2(0, 0);
	else
		direction = normalize(direction);

	UpdateLineBLensInfo();
}

void LineBLens::FinishConstructing(float* mv, float* pj, int winW, int winH)
{
	UpdateInfo(mv, pj, winW, winH);
	isConstructing = false;
}

std::vector<float2> LineBLens::GetCtrlPointsForRendering(float* mv, float* pj, int winW, int winH){
	std::vector<float2> res;
	if (isConstructing){
		res.push_back(ctrlPoint1Abs);
		res.push_back(ctrlPoint2Abs);
	}
	else{
		float2 center = GetCenterScreenPos(mv, pj, winW, winH);
		float2 ctrlPoint1 = center - direction*lSemiMajorAxis;
		float2 ctrlPoint2 = center + direction*lSemiMajorAxis;
		res.push_back(ctrlPoint1);
		res.push_back(ctrlPoint2);
		float2 minorDirection = make_float2(-direction.y, direction.x);
		float2 semiCtrlPoint1 = center - minorDirection*lSemiMinorAxis;
		float2 semiCtrlPoint2 = center + minorDirection*lSemiMinorAxis;
		res.push_back(semiCtrlPoint1);
		res.push_back(semiCtrlPoint2);
	}
	return res;
}


bool LineBLens::PointOnInnerBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) 
{
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);

	float2 toPoint = make_float2(_x, _y) - center;
	float disMajorAbs = std::abs(toPoint.x*direction.x + toPoint.y*direction.y);
	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinorAbs = std::abs(toPoint.x*minorDirection.x + toPoint.y*minorDirection.y);

	return (std::abs(disMajorAbs - lSemiMajorAxis) < eps_pixel && disMinorAbs <= lSemiMinorAxis)
		|| (std::abs(disMinorAbs - lSemiMinorAxis) < eps_pixel && disMajorAbs <= lSemiMajorAxis);
}


bool LineBLens::PointOnOuterBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) 
{
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);

	float2 toPoint = make_float2(_x, _y) - center;
	float disMajorAbs = std::abs(toPoint.x*direction.x + toPoint.y*direction.y);
	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinorAbs = std::abs(toPoint.x*minorDirection.x + toPoint.y*minorDirection.y);

	return (std::abs(disMajorAbs - lSemiMajorAxis) < eps_pixel && disMinorAbs > lSemiMinorAxis && disMinorAbs <= lSemiMinorAxis / focusRatio)
		|| (std::abs(disMinorAbs - lSemiMinorAxis / focusRatio) < eps_pixel && disMajorAbs <= lSemiMajorAxis);
}


bool LineBLens::PointOnObjectInnerBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH)
{
	return PointOnInnerBoundary(_x, _y, mv, pj, winW, winH);
}

bool LineBLens::PointOnObjectOuterBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH)
{
	return PointOnOuterBoundary(_x, _y, mv, pj, winW, winH);
}

void LineBLens::ChangeObjectLensSize(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)
{
	ChangeLensSize(_x, _y, _prex, _prey, mv, pj, winW, winH);
}

void LineBLens::ChangeObjectFocusRatio(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)
{
	ChangefocusRatio(_x, _y, _prex, _prey, mv, pj, winW, winH);
}

void LineBLens::ChangeLensSize(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)
{
	//float disThr = max(eps_pixel / 4, 10);
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);

	float2 toPoint = make_float2(_x, _y) - center;
	float disMajor = toPoint.x*direction.x + toPoint.y*direction.y;
	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinor = toPoint.x*minorDirection.x + toPoint.y*minorDirection.y;


	if (std::abs(disMajor - lSemiMajorAxis) < eps_pixel && std::abs(disMinor) <= lSemiMinorAxis){
		float2 ctrlPoint2 = center + direction*lSemiMajorAxis;
		float2 newctrlPoint2 = ctrlPoint2 + (make_float2(_x, _y) - make_float2(_prex, _prey));
		lSemiMajorAxis = length(newctrlPoint2 - center);
		direction = normalize(newctrlPoint2 - center);
	}
	else if (std::abs(-disMajor - lSemiMajorAxis) < eps_pixel && std::abs(disMinor) <= lSemiMinorAxis){
		float2 ctrlPoint1 = center - direction*lSemiMajorAxis;
		float2 newctrlPoint1 = ctrlPoint1 + (make_float2(_x, _y) - make_float2(_prex, _prey));
		lSemiMajorAxis = length(newctrlPoint1 - center);
		direction = normalize(center - newctrlPoint1);
	}
	else if (std::abs(disMinor - lSemiMinorAxis) < eps_pixel && std::abs(disMajor) <= lSemiMajorAxis){
		float2 minorCtrlPoint2 = center + minorDirection*lSemiMinorAxis;
		float2 newminorCtrlPoint2 = minorCtrlPoint2 + (make_float2(_x, _y) - make_float2(_prex, _prey));;
		lSemiMinorAxis = length(newminorCtrlPoint2 - center);

		float2 newmd = normalize(newminorCtrlPoint2 - center);
		float2 newd = make_float2(-newmd.y, newmd.x);
		direction = newd;
	}
	else if (std::abs(-disMinor - lSemiMinorAxis) < eps_pixel && std::abs(disMajor) <= lSemiMajorAxis){
		float2 minorCtrlPoint1 = center - minorDirection*lSemiMinorAxis;
		float2 newminorCtrlPoint1 = minorCtrlPoint1 + (make_float2(_x, _y) - make_float2(_prex, _prey));;
		lSemiMinorAxis = length(newminorCtrlPoint1 - center);
		
		float2 newmd = normalize(newminorCtrlPoint1 - center);
		float2 newd = make_float2(-newmd.y, newmd.x);

		direction = -newd;
	}
	UpdateLineBLensInfo();
	/*
	//only change size but not direction
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);

	float2 toPoint = make_float2(_x, _y) - center;
	float disMajorAbs = std::abs(toPoint.x*direction.x + toPoint.y*direction.y);
	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinorAbs = std::abs(toPoint.x*minorDirection.x + toPoint.y*minorDirection.y);

	if (std::abs(disMajorAbs - lSemiMajorAxis) < eps_pixel && disMinorAbs <= lSemiMinorAxis)
		lSemiMajorAxis = disMajorAbs;
	else if(std::abs(disMinorAbs - lSemiMinorAxis) < eps_pixel && disMajorAbs <= lSemiMajorAxis)
		lSemiMinorAxis = disMinorAbs;
		*/
}

void LineBLens::ChangefocusRatio(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)
{
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);

	float2 toPoint = make_float2(_x, _y) - center;
	float disMajorAbs = std::abs(toPoint.x*direction.x + toPoint.y*direction.y);
	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinorAbs = std::abs(toPoint.x*minorDirection.x + toPoint.y*minorDirection.y);

	if (std::abs(disMinorAbs - lSemiMinorAxis / focusRatio) < eps_pixel && disMajorAbs <= lSemiMajorAxis)
	{
		if (disMinorAbs > lSemiMinorAxis + eps_pixel + 1)
			focusRatio = lSemiMinorAxis / disMinorAbs;
	}
	UpdateLineBLensInfo();
}

/*
void LineBLens::ChangeDirection(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)
{
	//only rotate
	float disThr = max(eps_pixel / 4, 10);
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);

	float2 toPoint = make_float2(_x, _y) - center;
	float disMajorAbs = std::abs(toPoint.x*direction.x + toPoint.y*direction.y);
	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinorAbs = std::abs(toPoint.x*minorDirection.x + toPoint.y*minorDirection.y);

	if (std::abs(disMajorAbs - lSemiMajorAxis) < disThr && disMinorAbs <= disThr){
		float2 newd = normalize(make_float2(_x, _y) - center);
		if (dot(newd, direction) < 0)
			direction = -newd;
		else
			direction = newd;
	}
	else if (std::abs(disMinorAbs - lSemiMinorAxis) < disThr && disMajorAbs <= disThr){
		float2 newmd = normalize(make_float2(_x, _y) - center);
		float2 newd = make_float2(-newmd.y, newmd.x);
		if (dot(newd, direction) < 0)
			direction = -newd;
		else
			direction = newd;
	}
}
*/

void redistributePoints(std::vector<float2> & p)
{
	int n = p.size();
	if (n > 2){
		std::vector<float2> orip = p;
		float totalDis = 0;
		for (int i = 1; i < n; i++){
			totalDis += length(p[i] - p[i - 1]);
		}
		float segDis = totalDis/(n - 1);
		float curDis = 0;
		float2 curPos = orip[0];
		int nextId = 1;

		for (int i = 1; i < n - 1; i++){
			float targetDis = segDis*i;			
			while (curDis < targetDis){
				if (curDis + length(orip[nextId] - curPos) < targetDis){
					curDis = curDis + length(orip[nextId] - curPos);
					curPos = orip[nextId];
					nextId++;
				}
				else{
					curPos = curPos + normalize(orip[nextId] - curPos)*(targetDis - curDis);
					curDis = targetDis;
				}
			}
			p[i] = curPos;
		}
	}
}



void CurveBLens::FinishConstructing(float* mv, float* pj, int winW, int winH)
{
	if (numCtrlPoints >= 3){
		
		redistributePoints(ctrlPointsAbs);

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
		int end2ind = numCtrlPoints - end1ind - 1;
		if (end2ind >= 0 && end2ind < numCtrlPoints - 1){
			float dis = length(ctrlPointsAbs[numCtrlPoints - 1] - ctrlPointsAbs[end2ind]);
			float2 dir = normalize(ctrlPointsAbs[numCtrlPoints - 1] - ctrlPointsAbs[end2ind]);
			for (int i = end2ind + 1; i < numCtrlPoints - 1; i++){
				ctrlPointsAbs[i] = ctrlPointsAbs[end2ind] + dis*(i - end2ind) / end1ind*dir;
			}
		}
		//std::cout << numCtrlPoints << " " << end1ind << " " << end2ind << std::endl;


		//remove self intersection
		ctrlPointsAbs = removeSelfIntersection(ctrlPointsAbs, false);
		numCtrlPoints = ctrlPointsAbs.size();

		//compute center and relative positions
		float sumx = 0, sumy = 0;
		for (int ii = 0; ii < numCtrlPoints; ii++) {
			sumx += ctrlPointsAbs[ii].x, sumy += ctrlPointsAbs[ii].y;  //sum of std::absolute position
		}
		UpdateCenterByScreenPos(sumx / numCtrlPoints, sumy / numCtrlPoints, mv, pj, winW, winH);
		float2 center = GetCenterScreenPos(mv, pj, winW, winH);// make_float2(x, y);
		ctrlPoints.resize(numCtrlPoints);
		for (int ii = 0; ii < numCtrlPoints; ii++) {
			ctrlPoints[ii] = ctrlPointsAbs[ii] - center;
		}

		//optional: refine the control points shape and reduce the number, by the Bezier Curve
		//better strategy is to check methods about bezier down degree with minimum shape loss
		if (0){
			std::vector<float2> BezierSmapleOri = BezierSmaple(ctrlPoints);
			numCtrlPoints = numCtrlPoints / 2;
			ctrlPoints.resize(numCtrlPoints);
			for (int ii = 0; ii < numCtrlPoints; ii++) {
				ctrlPoints[ii] = BezierSmapleOri[ii * 2];
			}
		}

		//compute the BezierPoints used to draw the curve, as well as the normals
		BezierPoints = BezierSmaple(ctrlPoints);
		std::vector<float2> Q;
		std::vector<float> us;
		int numBP = BezierPoints.size();
		for (int i = 0; i < numBP - 1; i++){
			Q.push_back((BezierPoints[i + 1] - BezierPoints[i])*numBP);
		}
		for (int i = 0; i < numBP; i++){
			us.push_back(i*1.0 / (numBP-1));
		}
		std::vector<float2> BezierTangets;
		//see: http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html
		BezierTangets = BezierSmaple(Q,us);
		for (int i = 0; i < numBP; i++){
			BezierNormals.push_back(make_float2(-BezierTangets[i].y, BezierTangets[i].x));
		}
		
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

		////compute curveBLensInfo
		{
			curveBLensInfo.width = width;
			curveBLensInfo.focusRatio = focusRatio;

			curveBLensInfo.numBezierPoints = BezierPoints.size();
			for (int i = 0; i < BezierPoints.size(); i++){
				curveBLensInfo.BezierPoints[i] = BezierPoints[i];
			}

			curveBLensInfo.numPosPoints = subCtrlPointsPos.size();
			for (int i = 0; i < subCtrlPointsPos.size(); i++){
				curveBLensInfo.subCtrlPointsPos[i] = subCtrlPointsPos[i];
				curveBLensInfo.posOffsetCtrlPoints[i] = posOffsetCtrlPoints[i];
			}

			curveBLensInfo.numNegPoints = subCtrlPointsNeg.size();
			for (int i = 0; i < subCtrlPointsNeg.size(); i++){
				curveBLensInfo.subCtrlPointsNeg[i] = subCtrlPointsNeg[i];
				curveBLensInfo.negOffsetCtrlPoints[i] = negOffsetCtrlPoints[i];
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
	posOffsetCtrlPoints.clear(); //!!! maybe can improve performance by using std::vector<>::reserve

	dir = normalize(subCtrlPointsPos[1] - subCtrlPointsPos[0]);
	normal = make_float2(-dir.y, dir.x);
	posOffsetCtrlPoints.push_back(subCtrlPointsPos[0] + normal*outerWidth);

	for (int i = 1; i < np - 1; i++){
		dir = normalize(subCtrlPointsPos[i] - subCtrlPointsPos[i - 1]);
		normal = make_float2(-dir.y, dir.x);
		float2 dir2 = normalize(subCtrlPointsPos[i + 1] - subCtrlPointsPos[i]);
		float2 normal2 = make_float2(-dir.y, dir.x);
		float2 posCtrlPoint;
		intersectPoint(subCtrlPointsPos[i - 1] + normal*outerWidth, subCtrlPointsPos[i] + normal*outerWidth, subCtrlPointsPos[i] + normal2*outerWidth, subCtrlPointsPos[i + 1] + normal2*outerWidth, posCtrlPoint);
		posOffsetCtrlPoints.push_back(posCtrlPoint);
	}

	dir = normalize(subCtrlPointsPos[np - 1] - subCtrlPointsPos[np - 2]);
	normal = make_float2(-dir.y, dir.x);
	posOffsetCtrlPoints.push_back(subCtrlPointsPos[np - 1] + normal*outerWidth);
}

void CurveBLens::offsetControlPointsNeg()
{
	float2 normal;
	float2 dir;
		
	int nn = subCtrlPointsNeg.size();
	negOffsetCtrlPoints.clear();

	dir = normalize(subCtrlPointsNeg[1] - subCtrlPointsNeg[0]);
	normal = make_float2(-dir.y, dir.x);
	negOffsetCtrlPoints.push_back(subCtrlPointsNeg[0] - normal*outerWidth);

	for (int i = 1; i < nn - 1; i++){
		dir = normalize(subCtrlPointsNeg[i] - subCtrlPointsNeg[i - 1]);
		normal = make_float2(-dir.y, dir.x);
		float2 dir2 = normalize(subCtrlPointsNeg[i + 1] - subCtrlPointsNeg[i]);
		float2 normal2 = make_float2(-dir.y, dir.x);

		float2 negCtrlPoint;
		intersectPoint(subCtrlPointsNeg[i - 1] - normal*outerWidth, subCtrlPointsNeg[i] - normal*outerWidth, subCtrlPointsNeg[i] - normal2*outerWidth, subCtrlPointsNeg[i + 1] - normal2*outerWidth, negCtrlPoint);
		negOffsetCtrlPoints.push_back(negCtrlPoint);
	}

	dir = normalize(subCtrlPointsNeg[nn - 1] - subCtrlPointsNeg[nn - 2]);
	normal = make_float2(-dir.y, dir.x);
	negOffsetCtrlPoints.push_back(subCtrlPointsNeg[nn - 1] - normal*outerWidth);

}


void CurveBLens::computeBoundaryPos()
{
	int np = posOffsetCtrlPoints.size();
	int numberPosBezierPart = pow(2, refinedRoundPos);
	posOffsetBezierPoints.clear();
	for (int i = 0; i < numberPosBezierPart; i++){
		std::vector<float2> curPart(
			posOffsetCtrlPoints.begin() + (np - 1) / numberPosBezierPart*i,
			posOffsetCtrlPoints.begin() + (np - 1) / numberPosBezierPart*(i + 1) + 1);
		std::vector<float2> curBezierPart = BezierSmaple(curPart);
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
		std::vector<float2> curPart(
			negOffsetCtrlPoints.begin() + (nn - 1) / numberNegBezierPart*i,
			negOffsetCtrlPoints.begin() + (nn - 1) / numberNegBezierPart*(i + 1) + 1);
		std::vector<float2> curBezierPart = BezierSmaple(curPart);
		if (i == 0)
			negOffsetBezierPoints.insert(negOffsetBezierPoints.end(), curBezierPart.begin(), curBezierPart.end());
		else
			negOffsetBezierPoints.insert(negOffsetBezierPoints.end(), curBezierPart.begin() + 1, curBezierPart.end());
	}
}


void CurveBLens::computeRenderingBoundaryPos(std::vector<float2> &ret, int bezierSampleAccuracyRate)
{
	int np = posOffsetCtrlPoints.size();
	int numberPosBezierPart = pow(2, refinedRoundPos);
	ret.clear();
	for (int i = 0; i < numberPosBezierPart; i++){
		std::vector<float2> curPart(
			posOffsetCtrlPoints.begin() + (np - 1) / numberPosBezierPart*i,
			posOffsetCtrlPoints.begin() + (np - 1) / numberPosBezierPart*(i + 1) + 1);
		std::vector<float2> curBezierPart = BezierSmaple(curPart, bezierSampleAccuracyRate);
		if (i == 0)
			ret.insert(ret.end(), curBezierPart.begin(), curBezierPart.end());
		else
			ret.insert(ret.end(), curBezierPart.begin() + 1, curBezierPart.end());
	}

}

void CurveBLens::computeRenderingBoundaryNeg(std::vector<float2> &ret, int bezierSampleAccuracyRate)
{
	int nn = negOffsetCtrlPoints.size();
	int numberNegBezierPart = pow(2, refinedRoundNeg);
	ret.clear();
	for (int i = 0; i < numberNegBezierPart; i++){
		std::vector<float2> curPart(
			negOffsetCtrlPoints.begin() + (nn - 1) / numberNegBezierPart*i,
			negOffsetCtrlPoints.begin() + (nn - 1) / numberNegBezierPart*(i + 1) + 1);
		std::vector<float2> curBezierPart = BezierSmaple(curPart, bezierSampleAccuracyRate);
		if (i == 0)
			ret.insert(ret.end(), curBezierPart.begin(), curBezierPart.end());
		else
			ret.insert(ret.end(), curBezierPart.begin() + 1, curBezierPart.end());
	}
}

/*
std::vector<float2> CurveBLens::GetContour(float* mv, float* pj, int winW, int winH){
	std::vector<float2> ret;

	//may not use posOffsetBezierPoints directly since the num of points is too few
	if (posOffsetCtrlPoints.size()>2){

		float2 center = GetCenterScreenPos(mv, pj, winW, winH);// make_float2(x, y);
		int n = posOffsetBezierPoints.size();

		//!!! probably can be more precise with focusRatio!!!
		for (int ii = 0; ii < n; ii++){
			ret.push_back((subCtrlPointsPos[ii] + posOffsetBezierPoints[ii]) / 2.0 + center);
		}
		std::vector<float2> rettemp;
		int m = negOffsetBezierPoints.size();
		for (int ii = 0; ii < m; ii++){
			rettemp.push_back((subCtrlPointsNeg[ii] + negOffsetBezierPoints[ii]) / 2.0 + center);
		}

		std::reverse(rettemp.begin(), rettemp.end());
		ret.insert(ret.end(), rettemp.begin(), rettemp.end());

	}
	return ret;
}

std::vector<float2> CurveBLens::GetOuterContour(float* mv, float* pj, int winW, int winH){
	std::vector<float2> ret;
	if (posOffsetBezierPoints.size()>2){
		//if (!isConstructing && numCtrlPoints >= 3) {
		float2 center = GetCenterScreenPos(mv, pj, winW, winH);// make_float2(x, y);
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

*/

//do not use posOffsetBezierPoints directly since the num of points is too few
std::vector<float2> CurveBLens::GetContour(float* mv, float* pj, int winW, int winH){
	std::vector<float2> ret;

	if (!isConstructing && posOffsetCtrlPoints.size()>2){

		int segPrec = 4;
		std::vector<float2> bppos, bpneg;
		computeRenderingBoundaryPos(bppos, segPrec);
		computeRenderingBoundaryNeg(bpneg, segPrec);

		int numberPosBezierPart = pow(2, refinedRoundPos);
		int numberNegBezierPart = pow(2, refinedRoundNeg);
		std::vector<float2> centerCurvePos = BezierSmaple(ctrlPoints, segPrec*numberPosBezierPart);
		std::vector<float2> centerCurveNeg = BezierSmaple(ctrlPoints, segPrec*numberNegBezierPart);



		float2 center = GetCenterScreenPos(mv, pj, winW, winH);// make_float2(x, y);
		int n = centerCurvePos.size();
		for (int ii = 0; ii < n; ii++){
			ret.push_back((centerCurvePos[ii] + bppos[ii]) / 2.0 + center);
		}
		std::vector<float2> rettemp;
		int m = centerCurveNeg.size();
		for (int ii = 0; ii < m; ii++){
			rettemp.push_back((centerCurveNeg[ii] + bpneg[ii]) / 2.0 + center);
		}

		std::reverse(rettemp.begin(), rettemp.end());
		ret.insert(ret.end(), rettemp.begin(), rettemp.end());

	}
	return ret;
}


std::vector<float2> CurveBLens::GetOuterContour(float* mv, float* pj, int winW, int winH){
	std::vector<float2> ret;
	if (!isConstructing && posOffsetBezierPoints.size()>2){

		int segPrec = 4;
		std::vector<float2> bppos, bpneg;
		computeRenderingBoundaryPos(bppos, segPrec);
		computeRenderingBoundaryNeg(bpneg, segPrec);

		float2 center = GetCenterScreenPos(mv, pj, winW, winH);// make_float2(x, y);
		int n = bppos.size();
		for (int ii = 0; ii < n; ii++){
			ret.push_back(bppos[ii] + center);
		}
		std::vector<float2> rettemp;
		int m = bpneg.size();
		for (int ii = 0; ii < m; ii++){
			rettemp.push_back(bpneg[ii] + center);
		}

		std::reverse(rettemp.begin(), rettemp.end());
		ret.insert(ret.end(), rettemp.begin(), rettemp.end());

	}
	return ret;
}


std::vector<float2> CurveBLens::GetCtrlPointsForRendering(float* mv, float* pj, int winW, int winH){
	std::vector<float2> ret;
	if (isConstructing){
		for (int ii = 0; ii < numCtrlPoints; ii++) {
			ret.push_back(make_float2(ctrlPointsAbs[ii].x, ctrlPointsAbs[ii].y));
		}
	}
	else{
		float2 center = GetCenterScreenPos(mv, pj, winW, winH);//make_float2(x, y);
		int n = ctrlPoints.size();
		for (int ii = 0; ii < n; ii++) {
			ret.push_back(ctrlPoints[ii] + center);
		}
	}
	return ret;
}

std::vector<float2> CurveBLens::GetCenterLineForRendering(float* mv, float* pj, int winW, int winH){
	std::vector<float2> ret;
	if (isConstructing){
		for (int ii = 0; ii < numCtrlPoints; ii++) {
			ret.push_back(make_float2(ctrlPointsAbs[ii].x, ctrlPointsAbs[ii].y));
		}
	}
	else{
		float2 center = GetCenterScreenPos(mv, pj, winW, winH);//make_float2(x, y);
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
			if (std::abs(length(BezierPoints[i] - posOffsetBezierPoints[round(i *1.0 / (oriN - 1)*(posOBPsize - 1))]) - outerWidth) < outerWidth*offsetDisThr){
				//newRationalCtrlPoints.push_back(subCtrlPointsPos[i]);
			}
			else{
				convergedPos = false;
			}
		}
		if (!convergedPos){
			int m = subCtrlPointsPos.size();
			int numberPosBezierPart = pow(2, refinedRoundPos);

			std::vector<float2> newRCP;
			for (int i = 0; i < numberPosBezierPart; i++){
				std::vector<float2> curPart(
					subCtrlPointsPos.begin() + (m - 1) / numberPosBezierPart*i,
					subCtrlPointsPos.begin() + (m - 1) / numberPosBezierPart*(i + 1) + 1);
				std::vector<float2> refinedCurPart = BezierSubdivide(curPart, 1, 0.5);
				if (i == 0)
					newRCP.insert(newRCP.end(), refinedCurPart.begin(), refinedCurPart.end());
				else
					newRCP.insert(newRCP.end(), refinedCurPart.begin() + 1, refinedCurPart.end());

			}
			subCtrlPointsPos = newRCP;
			refinedRoundPos++;

			offsetControlPointsPos();
			computeBoundaryPos();

			std::cout << "refine for pos boundary once! count boundary points: " << subCtrlPointsPos.size() << std::endl;
		}
		else{
			std::cout << "pos converged!" << std::endl;
		}
	}


	//process the negative one
	{
		int oriN = BezierPoints.size();
		int negOBPsize = negOffsetBezierPoints.size();
		for (int i = 0; i < oriN; i++){
			//may need to be improved !!!
			if (std::abs(length(BezierPoints[i] - negOffsetBezierPoints[round(i *1.0 / (oriN - 1)*(negOBPsize - 1))]) - outerWidth) < outerWidth*offsetDisThr){
				//newRationalCtrlPoints.push_back(subCtrlPointsPos[i]);
			}
			else{
				convergedNeg = false;
			}
		}
		if (!convergedNeg){
			int m = subCtrlPointsNeg.size();
			int numberNegBezierPart = pow(2, refinedRoundNeg);

			std::vector<float2> newRCP;
			for (int i = 0; i < numberNegBezierPart; i++){
				std::vector<float2> curPart(
					subCtrlPointsNeg.begin() + (m - 1) / numberNegBezierPart*i,
					subCtrlPointsNeg.begin() + (m - 1) / numberNegBezierPart*(i + 1) + 1);
				std::vector<float2> refinedCurPart = BezierSubdivide(curPart, 1, 0.5);
				if (i == 0)
					newRCP.insert(newRCP.end(), refinedCurPart.begin(), refinedCurPart.end());
				else
					newRCP.insert(newRCP.end(), refinedCurPart.begin() + 1, refinedCurPart.end());

			}
			subCtrlPointsNeg = newRCP;
			refinedRoundNeg++;

			offsetControlPointsNeg();
			computeBoundaryNeg();

			std::cout << "refine for neg boundary once! count boundary points:" << subCtrlPointsNeg.size() << std::endl;
		}
		else{
			std::cout << "neg converged!" << std::endl;
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

std::vector<float2> CurveBLens::removeSelfIntersection(std::vector<float2> p, bool isDuplicating)
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
		std::cout << "no self intersection" << std::endl;
		return p;
	}
	else
	{
		std::cout << "did self intersection removal once!" << std::endl;
		std::vector<float2> newp;
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


std::vector<float2> CurveBLens::BezierOneSubdivide(std::vector<float2> p, std::vector<float2> poly1, std::vector<float2> poly2, float u)
{
	std::vector<float2> res;
	int n = p.size();
	if (n == 1)
	{
		res.insert(res.begin(), poly1.begin(), poly1.end());
		res.insert(res.end(), p[0]);
		res.insert(res.end(), poly2.begin(), poly2.end());
	}
	else if (p.size() > 1){
		poly1.push_back(p[0]);
		std::vector<float2> newpoly2;
		newpoly2.push_back(p[n - 1]);
		newpoly2.insert(newpoly2.end(), poly2.begin(), poly2.end());

		std::vector<float2> newp;
		for (int i = 0; i < n - 1; i++) {
			newp.push_back(p[i] + u*(p[i + 1] - p[i]));
		}
		res = BezierOneSubdivide(newp, poly1, newpoly2, u);

	}
	return res;

}

std::vector<float2> CurveBLens::BezierSubdivide(std::vector<float2> p, int m, float u)
{
	std::vector<float2> res;
	if (m == 1) {
		std::vector<float2> poly1(0), poly2(0);
		res = BezierOneSubdivide(p, poly1, poly2, u);
	}
	else {
		std::vector<float2> poly1(0), poly2(0);
		std::vector<float2> doubledp = BezierOneSubdivide(p, poly1, poly2, u);

		int n = (doubledp.size() + 1) / 2;
		std::vector<float2> newpLeft(doubledp.begin(), doubledp.begin() + n);
		std::vector<float2> newpRight(doubledp.begin() + n - 1, doubledp.end());

		std::vector<float2> respLeft = BezierSubdivide(newpLeft, m - 1, u);
		std::vector<float2> respRight = BezierSubdivide(newpRight, m - 1, u);

		res.insert(res.begin(), respLeft.begin(), respLeft.end());
		res.insert(res.end(), respRight.begin() + 1, respRight.end());
	}
	return res;
}

std::vector<float2> CurveBLens::BezierSmaple(std::vector<float2> p, int bezierSampleAccuracyRate)
{
	std::vector<float2> res;
	if (p.size() >= 2){
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

std::vector<float2> CurveBLens::BezierSmaple(std::vector<float2> p, std::vector<float> us)//for computing the tangent. bezierSampleAccuracyRate is always 1 in this function
{
	std::vector<float2> res;
	if (p.size() >= 2){
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
		
		//for (int ui = 0; ui <= n*bezierSampleAccuracyRate; ui++){
			//float u = ui*1.0 / (n*bezierSampleAccuracyRate);

		for (int ui = 0; ui <us.size(); ui++){
			float u = us[ui];
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




bool CurveBLens::PointInsideLens(int _x, int _y, float* mv, float* pj, int winW, int winH)
{
	if (isConstructing)
		return true;
	float2 screenCoord = make_float2(_x, _y);

	float2 center = GetCenterScreenPos(mv, pj, winW, winH); //make_float2(x, y);

	int numPosPoints = subCtrlPointsPos.size();
	int numNegPoints = subCtrlPointsNeg.size();

	//possible difference of the numPosPoints and numNegPoints makes the positive half and the negative half region do npt cover the whole lens region, according to current method
#define DifResAdjust 0.05

	{
		//float2* normals = curveLensCtrlPoints.normals;
		int numPosCP = subCtrlPointsPos.size();
		//float2* keyPoints = curveLensCtrlPoints.keyPoints;

		bool segmentNotFoundPos = true;
		int keySegmentId = -1;
		for (int ii = 0; ii < numPosCP - 1 && segmentNotFoundPos; ii++) {
			float2 toPoint = screenCoord - (center + subCtrlPointsPos[ii]);
			float2 dir = normalize(subCtrlPointsPos[ii + 1] - subCtrlPointsPos[ii]);
			float2 minorDir = make_float2(-dir.y, dir.x);
			float disMinor = toPoint.x*minorDir.x + toPoint.y*minorDir.y;
			if (disMinor < width && (disMinor >= 0 || (numPosPoints<numNegPoints && disMinor >= -width/focusRatio*DifResAdjust))){
			//if (disMinor >= 0 && disMinor < width){
				float2 ctrlPointAbsolute1 = center + subCtrlPointsPos[ii];
				float2 ctrlPointAbsolute2 = center + subCtrlPointsPos[ii + 1];

				float2 normal1 = normalize(posOffsetCtrlPoints[ii] - subCtrlPointsPos[ii]);
				float2 normal2 = normalize(posOffsetCtrlPoints[ii + 1] - subCtrlPointsPos[ii + 1]);

				//first check if screenCoord and ctrlPointAbsolute2 are at the same side of Line (ctrlPointAbsolute1, normals[ii])
				//then check if screenCoord and ctrlPointAbsolute1 are at the same side of Line (ctrlPointAbsolute2, normals[ii+1])

				if (((screenCoord.x - ctrlPointAbsolute1.x)*normal1.y - (screenCoord.y - ctrlPointAbsolute1.y)*normal1.x)
					*((ctrlPointAbsolute2.x - ctrlPointAbsolute1.x)*normal1.y - (ctrlPointAbsolute2.y - ctrlPointAbsolute1.y)*normal1.x)
					>= 0) {
					if (((screenCoord.x - ctrlPointAbsolute2.x)*normal2.y - (screenCoord.y - ctrlPointAbsolute2.y)*normal2.x)
						*((ctrlPointAbsolute1.x - ctrlPointAbsolute2.x)*normal2.y - (ctrlPointAbsolute1.y - ctrlPointAbsolute2.y)*normal2.x)
						>= 0) {
						segmentNotFoundPos = false;
					}
				}
			}
		}

		if (!segmentNotFoundPos)
			return true;
	}

	{
		int numNegCP = subCtrlPointsNeg.size();

		bool segmentNotFoundNeg = true;
		int keySegmentId = -1;
		for (int ii = 0; ii < numNegCP - 1 && segmentNotFoundNeg; ii++) {
			float2 toPoint = screenCoord - (center + subCtrlPointsNeg[ii]);
			float2 dir = normalize(subCtrlPointsNeg[ii + 1] - subCtrlPointsNeg[ii]);
			float2 minorDir = make_float2(-dir.y, dir.x);
			float disMinor = toPoint.x*minorDir.x + toPoint.y*minorDir.y;
			if (disMinor >-width && (disMinor <= 0 || (numPosPoints>numNegPoints && disMinor <= width/focusRatio*DifResAdjust))){
				//if (disMinor <= 0 && disMinor >- width){
				float2 ctrlPointAbsolute1 = center + subCtrlPointsNeg[ii];
				float2 ctrlPointAbsolute2 = center + subCtrlPointsNeg[ii + 1];

				float2 normal1 = normalize(negOffsetCtrlPoints[ii] - subCtrlPointsNeg[ii]);
				float2 normal2 = normalize(negOffsetCtrlPoints[ii + 1] - subCtrlPointsNeg[ii + 1]);

				//first check if screenCoord and ctrlPointAbsolute2 are at the same side of Line (ctrlPointAbsolute1, normals[ii])
				//then check if screenCoord and ctrlPointAbsolute1 are at the same side of Line (ctrlPointAbsolute2, normals[ii+1])

				if (((screenCoord.x - ctrlPointAbsolute1.x)*normal1.y - (screenCoord.y - ctrlPointAbsolute1.y)*normal1.x)
					*((ctrlPointAbsolute2.x - ctrlPointAbsolute1.x)*normal1.y - (ctrlPointAbsolute2.y - ctrlPointAbsolute1.y)*normal1.x)
					>= 0) {
					if (((screenCoord.x - ctrlPointAbsolute2.x)*normal2.y - (screenCoord.y - ctrlPointAbsolute2.y)*normal2.x)
						*((ctrlPointAbsolute1.x - ctrlPointAbsolute2.x)*normal2.y - (ctrlPointAbsolute1.y - ctrlPointAbsolute2.y)*normal2.x)
						>= 0) {
						segmentNotFoundNeg = false;
					}
				}
			}
		}

		return !segmentNotFoundNeg;
	}
}