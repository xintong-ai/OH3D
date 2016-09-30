#include "Lens.h"
#include "TransformFunc.h"
#include <iostream>
#include <algorithm>
#include <iostream>

float Lens::GetClipDepth(float* mv, float* pj)
{
	return Object2Clip(GetCenter(), mv, pj).z;
}

float2 Lens::GetCenterScreenPos(float* mv, float* pj, int winW, int winH)
{
	return Object2Screen(GetCenter(), mv, pj, winW, winH);
}

float3 Lens::UpdateCenterByScreenPos(int sx, int sy, float* mv, float* pj, int winW, int winH)
{
	float4 oriCenter = GetCenter();

	matrix4x4 invModelview, invProjection;
	invertMatrix(mv, &invModelview.v[0].x);
	invertMatrix(pj, &invProjection.v[0].x);
	float4 cenClip = Object2Clip(GetCenter(), mv, pj);
	float2 newClipXY = Screen2Clip(make_float2(sx, sy), winW, winH);
	float4 newClip = make_float4(newClipXY.x, newClipXY.y, cenClip.z, cenClip.w);
	float4 newObject = Clip2ObjectGlobal(newClip, &invModelview.v[0].x, &invProjection.v[0].x);
	SetCenter(make_float3(newObject));
	return make_float3(GetCenter() - oriCenter);
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
	dir3 = dir3 * (1.0f / length(dir3)) * v * (-0.05)* (0.05);
	SetCenter(make_float3(
		c.x + dir3.x,
		c.y + dir3.y,
		c.z + dir3.z));
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
	return abs(dis - objectRadius) < eps_dis;
}

bool CircleLens::PointOnObjectOuterBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH)
{
	float3 clickPoint = Compute3DPosByScreenPos(_x, _y, mv, pj, winW, winH);
	float eps_dis = objectRadius*0.1 / focusRatio;
	float dis = length(c - clickPoint);
	return abs(dis - objectRadius/focusRatio) < eps_dis;
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


bool LineLens::PointInsideLens(int _x, int _y, float* mv, float* pj, int winW, int winH) 
{
	//dot product of (_x-x, _y-y) and direction
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);
	float2 toPoint = make_float2(_x - center.x, _y - center.y);
	float disMajor = toPoint.x*direction.x + toPoint.y*direction.y;

	if (abs(disMajor) < lSemiMajorAxis) {
		float2 minorDirection = make_float2(-direction.y, direction.x);
		//dot product of (_x-x, _y-y) and minorDirection
		float disMinor = (_x - center.x)*minorDirection.x + (_y - center.y)*minorDirection.y;
		if (abs(disMinor) < lSemiMinorAxis)
			return true;
	}
	return false;
}


float3 LineLens::UpdateCenterByScreenPos(int sx, int sy, float* mv, float* pj, int winW, int winH)
{
	float4 oriCenter = GetCenter();

	matrix4x4 invModelview, invProjection;
	invertMatrix(mv, &invModelview.v[0].x);
	invertMatrix(pj, &invProjection.v[0].x);

	float2 oriCenter_screen = Object2Screen(oriCenter, mv, pj, winW, winH);
	ctrlPoint1Abs = ctrlPoint1Abs + make_float2(sx, sy) - oriCenter_screen;
	ctrlPoint2Abs = ctrlPoint2Abs + make_float2(sx, sy) - oriCenter_screen;

	float4 cenClip = Object2Clip(GetCenter(), mv, pj);
	float2 newClipXY = Screen2Clip(make_float2(sx, sy), winW, winH);
	float4 newClip = make_float4(newClipXY.x, newClipXY.y, cenClip.z, cenClip.w);
	float4 newObject = Clip2ObjectGlobal(newClip, &invModelview.v[0].x, &invProjection.v[0].x);
	SetCenter(make_float3(newObject));
	return make_float3(GetCenter() - oriCenter);
}


std::vector<float2> LineLens::GetContour(float* mv, float* pj, int winW, int winH)
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

std::vector<float2> LineLens::GetOuterContour(float* mv, float* pj, int winW, int winH)
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
void LineLens::UpdateLineLensInfo()
{
	lineLensInfo.lSemiMajorAxis = lSemiMajorAxis;
	lineLensInfo.lSemiMinorAxis = lSemiMinorAxis;
	lineLensInfo.direction = direction;
	lineLensInfo.focusRatio = focusRatio;
}

void LineLens::UpdateInfoFromCtrlPoints(float* mv, float* pj, int winW, int winH)
{
	UpdateCenterByScreenPos((ctrlPoint1Abs.x + ctrlPoint2Abs.x) / 2.0, (ctrlPoint1Abs.y + ctrlPoint2Abs.y) / 2.0, mv, pj, winW, winH);

	direction = ctrlPoint2Abs - ctrlPoint1Abs;

	lSemiMajorAxis = length(direction) / 2;
	lSemiMinorAxis = lSemiMajorAxis / axisRatio;

	if (lSemiMajorAxis < 0.000001)
		direction = make_float2(0, 0);
	else
		direction = normalize(direction);

	UpdateLineLensInfo();
}

void LineLens::FinishConstructing(float* mv, float* pj, int winW, int winH)
{
	UpdateInfoFromCtrlPoints(mv, pj, winW, winH);
	isConstructing = false;
}

std::vector<float2> LineLens::GetCtrlPointsForRendering(float* mv, float* pj, int winW, int winH){
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


bool LineLens::PointOnInnerBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) 
{
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);

	float2 toPoint = make_float2(_x, _y) - center;
	float disMajorAbs = abs(toPoint.x*direction.x + toPoint.y*direction.y);
	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinorAbs = abs(toPoint.x*minorDirection.x + toPoint.y*minorDirection.y);

	return (abs(disMajorAbs - lSemiMajorAxis) < eps_pixel && disMinorAbs <= lSemiMinorAxis)
		|| (abs(disMinorAbs - lSemiMinorAxis) < eps_pixel && disMajorAbs <= lSemiMajorAxis);
}


bool LineLens::PointOnOuterBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH) 
{
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);

	float2 toPoint = make_float2(_x, _y) - center;
	float disMajorAbs = abs(toPoint.x*direction.x + toPoint.y*direction.y);
	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinorAbs = abs(toPoint.x*minorDirection.x + toPoint.y*minorDirection.y);

	return (abs(disMajorAbs - lSemiMajorAxis) < eps_pixel && disMinorAbs > lSemiMinorAxis && disMinorAbs <= lSemiMinorAxis / focusRatio)
		|| (abs(disMinorAbs - lSemiMinorAxis / focusRatio) < eps_pixel && disMajorAbs <= lSemiMajorAxis);
}



void LineLens::ChangeLensSize(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)
{
	//float disThr = max(eps_pixel / 4, 10);
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);

	float2 toPoint = make_float2(_x, _y) - center;
	float disMajor = toPoint.x*direction.x + toPoint.y*direction.y;
	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinor = toPoint.x*minorDirection.x + toPoint.y*minorDirection.y;


	if (abs(disMajor - lSemiMajorAxis) < eps_pixel && abs(disMinor) <= lSemiMinorAxis){
		float2 ctrlPoint2 = center + direction*lSemiMajorAxis;
		float2 newctrlPoint2 = ctrlPoint2 + (make_float2(_x, _y) - make_float2(_prex, _prey));
		lSemiMajorAxis = length(newctrlPoint2 - center);
		direction = normalize(newctrlPoint2 - center);
	}
	else if (abs(-disMajor - lSemiMajorAxis) < eps_pixel && abs(disMinor) <= lSemiMinorAxis){
		float2 ctrlPoint1 = center - direction*lSemiMajorAxis;
		float2 newctrlPoint1 = ctrlPoint1 + (make_float2(_x, _y) - make_float2(_prex, _prey));
		lSemiMajorAxis = length(newctrlPoint1 - center);
		direction = normalize(center - newctrlPoint1);
	}
	else if (abs(disMinor - lSemiMinorAxis) < eps_pixel && abs(disMajor) <= lSemiMajorAxis){
		float2 minorCtrlPoint2 = center + minorDirection*lSemiMinorAxis;
		float2 newminorCtrlPoint2 = minorCtrlPoint2 + (make_float2(_x, _y) - make_float2(_prex, _prey));;
		lSemiMinorAxis = length(newminorCtrlPoint2 - center);

		float2 newmd = normalize(newminorCtrlPoint2 - center);
		float2 newd = make_float2(-newmd.y, newmd.x);
		direction = newd;
	}
	else if (abs(-disMinor - lSemiMinorAxis) < eps_pixel && abs(disMajor) <= lSemiMajorAxis){
		float2 minorCtrlPoint1 = center - minorDirection*lSemiMinorAxis;
		float2 newminorCtrlPoint1 = minorCtrlPoint1 + (make_float2(_x, _y) - make_float2(_prex, _prey));;
		lSemiMinorAxis = length(newminorCtrlPoint1 - center);
		
		float2 newmd = normalize(newminorCtrlPoint1 - center);
		float2 newd = make_float2(-newmd.y, newmd.x);

		direction = -newd;
	}
	ctrlPoint2Abs = center + direction*lSemiMajorAxis;
	ctrlPoint1Abs = center - direction*lSemiMajorAxis;

	UpdateLineLensInfo();
}

void LineLens::ChangefocusRatio(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)
{
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);

	float2 toPoint = make_float2(_x, _y) - center;
	float disMajorAbs = abs(toPoint.x*direction.x + toPoint.y*direction.y);
	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinorAbs = abs(toPoint.x*minorDirection.x + toPoint.y*minorDirection.y);

	if (abs(disMinorAbs - lSemiMinorAxis / focusRatio) < eps_pixel && disMajorAbs <= lSemiMajorAxis)
	{
		if (disMinorAbs > lSemiMinorAxis + eps_pixel + 1)
			focusRatio = lSemiMinorAxis / disMinorAbs;
	}

	ctrlPoint2Abs = center + direction*lSemiMajorAxis;
	ctrlPoint1Abs = center - direction*lSemiMajorAxis;
	UpdateLineLensInfo();
}



///////////////////////// LineLens3D ///////////////////////////

std::vector<float2> LineLens3D::GetContour(float* mv, float* pj, int winW, int winH)
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

std::vector<float2> LineLens3D::GetOuterContour(float* mv, float* pj, int winW, int winH)
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



void LineLens3D::UpdateLineLensGlobalInfoFromScreenInfo(int winWidth, int winHeight, float _mv[16], float _pj[16], float3 dataMin, float3 dataMax)
{
	//this function works correctly when the projection info of the front face has been set
	//the computation is done by using ctrlPoint1Abs, ctrlPoint2Abs, and the previous center to provide the depth value

	float _invmv[16];
	float _invpj[16];
	invertMatrix(_pj, _invpj);
	invertMatrix(_mv, _invmv);

	float3 tempCtrlPoint1_Camera = make_float3(Clip2Camera(make_float4(make_float3(Screen2Clip(ctrlPoint1Abs, winWidth, winHeight), -1.0f), 1.0f), _invpj));
	float3 tempCtrlPoint2_Camera = make_float3(Clip2Camera(make_float4(make_float3(Screen2Clip(ctrlPoint2Abs, winWidth, winHeight), -1.0f), 1.0f), _invpj));

	float3 camera_Camera = make_float3(0, 0, 0);

	float3 dirSide1 = normalize(tempCtrlPoint1_Camera - camera_Camera);
	float3 dirSide2 = normalize(tempCtrlPoint2_Camera - camera_Camera);
	float3 lensDir_camera = normalize(dirSide1 + dirSide2);

	//https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
	//screenCenter_camera = camera_Camera + t1*lensDir_camera
	//t1=|v2 x v1|/(v2 dot v3)
	//v1 = lensDir_camera - tempCtrlPoint1_Camera
	//v2 = tempCtrlPoint2_Camera - tempCtrlPoint1_Camera
	//v3 = ...
	float3 v1 = camera_Camera - tempCtrlPoint1_Camera;
	float3 v2 = tempCtrlPoint2_Camera - tempCtrlPoint1_Camera;
	float3 v3 = normalize(cross(lensDir_camera, cross(lensDir_camera, v2)));
	float t1 = length(cross(v2, v1)) / abs(dot(v2, v3));
	float3 screenCenter_camera = camera_Camera + t1*lensDir_camera;
	float4 oldcenterclip = Object2Clip(make_float4(c,1.0), _mv, _pj);
	float3 lensCen = make_float3(Clip2ObjectGlobal(make_float4(make_float3(GetXY(Camera2ClipGlobal(make_float4(screenCenter_camera, 1.0), _pj)), oldcenterclip.z), 1.0), _invmv, _invpj));
	
	c = lensCen;

	float3 cameraObj = make_float3(Camera2Object(make_float4(0, 0, 0, 1), _invmv));

	lensDir = normalize(cameraObj - lensCen);

	float volumeCornerx[2] = { dataMin.x, dataMax.x }, volumeCornery[2] = { dataMin.y, dataMax.y }, volumeCornerz[2] = { dataMin.z, dataMax.z };
	float rz1, rz2;//at the direction lensDir shooting from lensCenter, the range to cover the whole region
	rz1 = FLT_MAX, rz2 = -FLT_MAX;
	//currently computing r1 and r2 aggressively. cam be more improved later
	for (int k = 0; k < 2; k++){
		for (int j = 0; j < 2; j++){
			for (int i = 0; i < 2; i++){
				float3 dir = make_float3(volumeCornerx[i], volumeCornery[j], volumeCornerz[k]) - lensCen;
				float curz = dot(dir, lensDir);
				if (curz < rz1)
					rz1 = curz;
				if (curz > rz2)
					rz2 = curz;
			}
		}
	}
	float zdifori = rz2 - rz1;
	rz2 = rz2 + zdifori*0.01;
	rz1 = rz1 - zdifori*0.01;  //to avoid numerical error

	frontBaseCenter = lensCen + rz2*lensDir;
	estMeshBottomCenter = lensCen + rz1*lensDir;

	float3 tempCtrlPoint1_obj = make_float3(Camera2Object(make_float4(tempCtrlPoint1_Camera, 1.0f), _invmv));
	float3 tempCtrlPoint2_obj = make_float3(Camera2Object(make_float4(tempCtrlPoint2_Camera, 1.0f), _invmv));

	minorAxisGlobal = normalize(cross(lensDir, tempCtrlPoint2_obj - tempCtrlPoint1_obj));
	majorAxisGlobal = normalize(cross(minorAxisGlobal, lensDir));
	

	//https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
	//intersection_point = cameraObj + x* d
	//d = normalize(tempCtrlPoint2_obj-cameraObj)
	//t2=(v1 dot v3)/(v2 dot v3)
	//v1 = cameraObj - frontBaseCenter
	//v2 = majorAxisGlobal
	//v3 = ...

	float3 d = normalize(tempCtrlPoint2_obj - cameraObj);
	v1 = cameraObj - frontBaseCenter;
	v2 = majorAxisGlobal;
	v3 = normalize(cross(d, cross(d, v2)));
	lSemiMajorAxisGlobal = dot(v1, v3) / dot(v2, v3);
	lSemiMinorAxisGlobal = lSemiMajorAxisGlobal / axisRatio;

	//cannot compute endPointSemiMinorAxisGlobal in a similar way to endPointSemiMajorAxisGlobal, which will cause non-perpendicular in global space
};

void LineLens3D::UpdateLineLensGlobalInfoFrom3DSegment(int winWidth, int winHeight, float _mv[16], float _pj[16], float3 dataMin, float3 dataMax)
{
	float3 lensCen = (ctrlPoint3D1 + ctrlPoint3D2) / 2;

	SetCenter(lensCen);

	float _invmv[16];
	float _invpj[16];
	invertMatrix(_pj, _invpj);
	invertMatrix(_mv, _invmv);

	float3 cameraObj = make_float3(Camera2Object(make_float4(0, 0, 0, 1), _invmv));

	lensDir = make_float3(
		cameraObj.x - lensCen.x,
		cameraObj.y - lensCen.y,
		cameraObj.z - lensCen.z);
	lensDir = normalize(lensDir);

	float3 drawingSeg = ctrlPoint3D2 - ctrlPoint3D1;
	minorAxisGlobal = cross(lensDir, drawingSeg);
	if (length(minorAxisGlobal) < 0.00001){
		std::cerr << "3d lens construction fail";
		exit(0);
	}
	minorAxisGlobal = normalize(minorAxisGlobal);

	majorAxisGlobal = normalize(cross(minorAxisGlobal, lensDir));

	float volumeCornerx[2] = { dataMin.x, dataMax.x }, volumeCornery[2] = { dataMin.y, dataMax.y }, volumeCornerz[2] = { dataMin.z, dataMax.z };
	float rz1, rz2;//at the direction lensDir shooting from lensCenter, the range to cover the whole region
	rz1 = FLT_MAX, rz2 = -FLT_MAX;
	//currently computing r1 and r2 aggressively. cam be more improved later
	for (int k = 0; k < 2; k++){
		for (int j = 0; j < 2; j++){
			for (int i = 0; i < 2; i++){
				float3 dir = make_float3(volumeCornerx[i], volumeCornery[j], volumeCornerz[k]) - lensCen;
				float curz = dot(dir, lensDir);
				if (curz < rz1)
					rz1 = curz;
				if (curz > rz2)
					rz2 = curz;
			}
		}
	}
	float zdifori = rz2 - rz1;
	rz2 = rz2 + zdifori*0.01;
	rz1 = rz1 - zdifori*0.01;  //to avoid numerical error

	frontBaseCenter = lensCen + rz2*lensDir;
	estMeshBottomCenter = lensCen + rz1*lensDir;
	lSemiMajorAxisGlobal = dot(drawingSeg, majorAxisGlobal) / 2;
	lSemiMinorAxisGlobal = lSemiMajorAxisGlobal / axisRatio;
	ctrlPoint3D1 = c - majorAxisGlobal*lSemiMajorAxisGlobal;
	ctrlPoint3D2 = c + majorAxisGlobal*lSemiMajorAxisGlobal;
}



void LineLens3D::UpdateLineLensGlobalInfo(int winWidth, int winHeight, float _mv[16], float _pj[16], float3 dataMin, float3 dataMax)
{
	if (isConstructedFromLeap){
		UpdateLineLensGlobalInfoFrom3DSegment(winWidth, winHeight, _mv, _pj, dataMin, dataMax);
	}
	else{
		UpdateLineLensGlobalInfoFromScreenInfo(winWidth, winHeight, _mv, _pj, dataMin, dataMax);
	}
}


void LineLens3D::FinishConstructing(float* _mv, float* _pj, int winW, int winH, float3 dataMin, float3 dataMax)
{
	isConstructing = false;
}

void LineLens3D::FinishConstructing3D(float* _mv, float* _pj, int winW, int winH, float3 dataMin, float3 dataMax)
{
	isConstructing = false;
}


bool LineLens3D::PointInsideLens(int _x, int _y, float* mv, float* pj, int winW, int winH)
{
	//dot product of (_x-x, _y-y) and direction
	float2 direction = normalize(ctrlPoint2Abs - ctrlPoint1Abs);
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);
	float2 toPoint = make_float2(_x - center.x, _y - center.y);
	float disMajor = toPoint.x*direction.x + toPoint.y*direction.y;
	
	if (abs(disMajor) < length(ctrlPoint2Abs-ctrlPoint1Abs)/2) //may not be precise
	{
		float2 minorDirection = make_float2(-direction.y, direction.x);
		//dot product of (_x-x, _y-y) and minorDirection
		float disMinor = dot(toPoint, minorDirection);
		if (abs(disMinor) < length(ctrlPoint2Abs - ctrlPoint1Abs) / 2/ axisRatio/focusRatio)
			return true;
	}
	return false;
}

bool LineLens3D::PointOnOuterBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH)
{
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);
	float2 toPoint = make_float2(_x, _y) - center;
	
	float2 direction = normalize(ctrlPoint2Abs - ctrlPoint1Abs);
	float disMajorAbs = abs(toPoint.x*direction.x + toPoint.y*direction.y);

	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinor = toPoint.x*minorDirection.x + toPoint.y*minorDirection.y;

	float3 cp1 = frontBaseCenter - majorAxisGlobal*lSemiMajorAxisGlobal;
	//float3 cp2 = frontBaseCenter + majorAxisGlobal*lSemiMajorAxisGlobal;
	float3 cpm1 = cp1 - minorAxisGlobal/focusRatio*lSemiMinorAxisGlobal;
	float3 cpm2 = cp1 + minorAxisGlobal / focusRatio*lSemiMinorAxisGlobal;

	//float2 cp1_screen = Object2Screen(make_float4(cp1, 1.0f), mv, pj, winW, winH);
	//float2 cp2_screen = Object2Screen(make_float4(cp2, 1.0f), mv, pj, winW, winH);
	float2 cp1_screen = ctrlPoint1Abs;
	float2 cp2_screen = ctrlPoint2Abs;
	float2 cpm1_screen = Object2Screen(make_float4(cpm1, 1.0f), mv, pj, winW, winH);
	float2 cpm2_screen = Object2Screen(make_float4(cpm2, 1.0f), mv, pj, winW, winH);


	return (disMajorAbs <abs(dot(cp1_screen - center, direction)) &&
		disMajorAbs < abs(dot(cp2_screen - center, direction)) &&
		(abs(disMinor - dot(cpm1_screen - center, minorDirection)) < eps_pixel ||
		abs(disMinor - dot(cpm2_screen - center, minorDirection)) < eps_pixel));
}

void LineLens3D::ChangefocusRatio(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)
{
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);
	float2 toPoint = make_float2(_x, _y) - center;

	float2 direction = normalize(ctrlPoint2Abs - ctrlPoint1Abs);
	float disMajorAbs = abs(toPoint.x*direction.x + toPoint.y*direction.y);

	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinor = toPoint.x*minorDirection.x + toPoint.y*minorDirection.y;

	float3 cp1 = frontBaseCenter - majorAxisGlobal*lSemiMajorAxisGlobal;
	//float3 cp2 = frontBaseCenter + majorAxisGlobal*lSemiMajorAxisGlobal;
	float3 cpm1 = cp1 - minorAxisGlobal / focusRatio*lSemiMinorAxisGlobal;
	float3 cpm2 = cp1 + minorAxisGlobal / focusRatio*lSemiMinorAxisGlobal;

	//float2 cp1_screen = Object2Screen(make_float4(cp1, 1.0f), mv, pj, winW, winH);
	//float2 cp2_screen = Object2Screen(make_float4(cp2, 1.0f), mv, pj, winW, winH);
	float2 cp1_screen = ctrlPoint1Abs;
	float2 cp2_screen = ctrlPoint2Abs;
	float2 cpm1_screen = Object2Screen(make_float4(cpm1, 1.0f), mv, pj, winW, winH);
	float2 cpm2_screen = Object2Screen(make_float4(cpm2, 1.0f), mv, pj, winW, winH);


	if (disMajorAbs < abs(dot(cp1_screen - center, direction)) &&
		disMajorAbs < abs(dot(cp2_screen - center, direction)))
	{
		if (abs(disMinor - dot(cpm1_screen - center, minorDirection)) < eps_pixel)
			
		{
			float3 cpm1Inner = cp1 - minorAxisGlobal*lSemiMinorAxisGlobal;
			float2 cpm1Inner_screen = Object2Screen(make_float4(cpm1Inner, 1.0f), mv, pj, winW, winH);
			focusRatio = abs(dot(cpm1Inner_screen - center, minorDirection) / disMinor);
		}
		else if (abs(disMinor - dot(cpm2_screen - center, minorDirection)) < eps_pixel)
		{
			float3 cpm2Inner = cp1 + minorAxisGlobal*lSemiMinorAxisGlobal;
			float2 cpm2Inner_screen = Object2Screen(make_float4(cpm2Inner, 1.0f), mv, pj, winW, winH);
			focusRatio = abs(dot(cpm2Inner_screen - center, minorDirection) / disMinor);
		}
	}
}

bool LineLens3D::PointOnInnerBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH)
{
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);
	float2 toPoint = make_float2(_x, _y) - center;

	float2 direction = normalize(ctrlPoint2Abs - ctrlPoint1Abs);
	float disMajor = toPoint.x*direction.x + toPoint.y*direction.y;

	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinorAbs = (toPoint.x*minorDirection.x + toPoint.y*minorDirection.y);

	float3 cp1 = frontBaseCenter - majorAxisGlobal*lSemiMajorAxisGlobal;
	//float3 cp2 = frontBaseCenter + majorAxisGlobal*lSemiMajorAxisGlobal;
	float3 cpm1 = cp1 - minorAxisGlobal / focusRatio*lSemiMinorAxisGlobal;
	float3 cpm2 = cp1 + minorAxisGlobal / focusRatio*lSemiMinorAxisGlobal;

	//float2 cp1_screen = Object2Screen(make_float4(cp1, 1.0f), mv, pj, winW, winH);
	//float2 cp2_screen = Object2Screen(make_float4(cp2, 1.0f), mv, pj, winW, winH);
	float2 cp1_screen = ctrlPoint1Abs;
	float2 cp2_screen = ctrlPoint2Abs;
	float2 cpm1_screen = Object2Screen(make_float4(cpm1, 1.0f), mv, pj, winW, winH);
	float2 cpm2_screen = Object2Screen(make_float4(cpm2, 1.0f), mv, pj, winW, winH);
	
	return (disMinorAbs < abs(dot(cpm1_screen - center, minorDirection)) &&
		disMinorAbs < abs(dot(cpm2_screen - center, minorDirection)) &&
		(abs(disMajor - dot(cp1_screen - center, direction)) < eps_pixel ||
		abs(disMajor - dot(cp2_screen - center, direction)) < eps_pixel));
}


void LineLens3D::ChangeLensSize(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)
{
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);
	float2 toPoint = make_float2(_x, _y) - center;

	float2 direction = normalize(ctrlPoint2Abs - ctrlPoint1Abs);
	float disMajor = toPoint.x*direction.x + toPoint.y*direction.y;

	float2 minorDirection = make_float2(-direction.y, direction.x);
	float disMinorAbs = (toPoint.x*minorDirection.x + toPoint.y*minorDirection.y);

	float3 cp1 = frontBaseCenter - majorAxisGlobal*lSemiMajorAxisGlobal;
	//float3 cp2 = frontBaseCenter + majorAxisGlobal*lSemiMajorAxisGlobal;
	float3 cpm1 = cp1 - minorAxisGlobal / focusRatio*lSemiMinorAxisGlobal;
	float3 cpm2 = cp1 + minorAxisGlobal / focusRatio*lSemiMinorAxisGlobal;

	//float2 cp1_screen = Object2Screen(make_float4(cp1, 1.0f), mv, pj, winW, winH);
	//float2 cp2_screen = Object2Screen(make_float4(cp2, 1.0f), mv, pj, winW, winH);
	float2 cp1_screen = ctrlPoint1Abs;
	float2 cp2_screen = ctrlPoint2Abs;
	float2 cpm1_screen = Object2Screen(make_float4(cpm1, 1.0f), mv, pj, winW, winH);
	float2 cpm2_screen = Object2Screen(make_float4(cpm2, 1.0f), mv, pj, winW, winH);

	if (disMinorAbs < abs(dot(cpm1_screen - center, minorDirection)) &&
		disMinorAbs < abs(dot(cpm2_screen - center, minorDirection))){
		if (abs(disMajor - dot(cp1_screen - center, direction)) < eps_pixel){
			ctrlPoint1Abs = ctrlPoint1Abs + (make_float2(_x, _y) - make_float2(_prex, _prey));
		}
		else if (abs(disMajor - dot(cp2_screen - center, direction)) < eps_pixel){
			ctrlPoint2Abs = ctrlPoint2Abs + (make_float2(_x, _y) - make_float2(_prex, _prey));
		}
	}
}



bool LineLens3D::PointOnObjectInnerBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH)
{
	return false; //implement later
}

bool LineLens3D::PointOnObjectOuterBoundary(int _x, int _y, float* mv, float* pj, int winW, int winH)
{
	return false; //implement later
}

void LineLens3D::ChangeObjectLensSize(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)
{
	ChangeLensSize(_x, _y, _prex, _prey, mv, pj, winW, winH);
}

void LineLens3D::ChangeObjectFocusRatio(int _x, int _y, int _prex, int _prey, float* mv, float* pj, int winW, int winH)
{
	ChangefocusRatio(_x, _y, _prex, _prey, mv, pj, winW, winH);
}



std::vector<float3> LineLens3D::GetOuterContourCenterFace()
{
	std::vector<float3> ret;
	float3 cp1 = c - majorAxisGlobal*lSemiMajorAxisGlobal;
	float3 cp2 = c + majorAxisGlobal*lSemiMajorAxisGlobal;

	ret.push_back(cp1 - minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio);
	ret.push_back(cp2 - minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio);
	ret.push_back(cp2 + minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio);
	ret.push_back(cp1 + minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio);
	return ret;
}

std::vector<float3> LineLens3D::GetOuterContourBackFace()
{
	std::vector<float3> ret;
	float3 cp1 = estMeshBottomCenter - majorAxisGlobal*lSemiMajorAxisGlobal;
	float3 cp2 = estMeshBottomCenter + majorAxisGlobal*lSemiMajorAxisGlobal;

	ret.push_back(cp1 - minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio);
	ret.push_back(cp2 - minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio);
	ret.push_back(cp2 + minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio);
	ret.push_back(cp1 + minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio);
	return ret;
}

std::vector<float3> LineLens3D::GetOuterContourFrontFace()
{
	std::vector<float3> ret;
	float3 cp1 = frontBaseCenter - majorAxisGlobal*lSemiMajorAxisGlobal;
	float3 cp2 = frontBaseCenter + majorAxisGlobal*lSemiMajorAxisGlobal;

	ret.push_back(cp1 - minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio);
	ret.push_back(cp2 - minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio);
	ret.push_back(cp2 + minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio);
	ret.push_back(cp1 + minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio);
	return ret;
}

std::vector<float3> LineLens3D::GetIncisionFront()
{
	std::vector<float3> ret;
	ret.push_back(frontBaseCenter - majorAxisGlobal*lSemiMajorAxisGlobal);
	ret.push_back(frontBaseCenter + majorAxisGlobal*lSemiMajorAxisGlobal);
	return ret;
}

std::vector<float3> LineLens3D::GetIncisionCenter()
{
	std::vector<float3> ret;
	//if (isConstructing && isConstructedFromLeap)
	ret.push_back(c - majorAxisGlobal*lSemiMajorAxisGlobal);
	ret.push_back(c + majorAxisGlobal*lSemiMajorAxisGlobal);
	return ret;
}

void LineLens3D::UpdateCtrlPoints(float* mv, float* pj, int winW, int winH)
{
	ctrlPoint1Abs = Object2Screen(make_float4(frontBaseCenter - majorAxisGlobal*lSemiMajorAxisGlobal,1.0), mv, pj, winW, winH);
	ctrlPoint2Abs = Object2Screen(make_float4(frontBaseCenter + majorAxisGlobal*lSemiMajorAxisGlobal, 1.0), mv, pj, winW, winH);
}

std::vector<float3> LineLens3D::Get3DCoordOfCtrlPoints(float* mv, float* pj, int winW, int winH)
{
	float _invmv[16];
	float _invpj[16];
	invertMatrix(pj, _invpj);
	invertMatrix(mv, _invmv);

	std::vector<float3> res;
	//if (isConstructing){
	res.push_back(make_float3(Clip2ObjectGlobal(make_float4(make_float3(Screen2Clip(ctrlPoint1Abs, winW, winH), -1), 1.0), _invmv, _invpj)));
	res.push_back(make_float3(Clip2ObjectGlobal(make_float4(make_float3(Screen2Clip(ctrlPoint2Abs, winW, winH), -1), 1.0), _invmv, _invpj)));
	//}
	//else{
	//	float2 center = GetCenterScreenPos(mv, pj, winW, winH);
	//	float2 ctrlPoint1 = center - direction*lSemiMajorAxis;
	//	float2 ctrlPoint2 = center + direction*lSemiMajorAxis;
	//	res.push_back(ctrlPoint1);
	//	res.push_back(ctrlPoint2);
	//	float2 minorDirection = make_float2(-direction.y, direction.x);
	//	float2 semiCtrlPoint1 = center - minorDirection*lSemiMinorAxis;
	//	float2 semiCtrlPoint2 = center + minorDirection*lSemiMinorAxis;
	//	res.push_back(semiCtrlPoint1);
	//	res.push_back(semiCtrlPoint2);
	//}
	return res;
}


std::vector<float3> LineLens3D::GetCtrlPoints3DForRendering(float* mv, float* pj, int winW, int winH)
{
	std::vector<float3> res;
	if (isConstructing){
		res.push_back(ctrlPoint3D1);
		res.push_back(ctrlPoint3D2);
	}
	else{
		res.push_back(c + majorAxisGlobal*lSemiMajorAxisGlobal);
		res.push_back(c - majorAxisGlobal*lSemiMajorAxisGlobal);
	}
	return res;
}

bool LineLens3D::PointOnLensCenter(int _x, int _y, float* mv, float* pj, int winW, int winH) {
	float2 center = GetCenterScreenPos(mv, pj, winW, winH);
	float dis = length(make_float2(_x, _y) - center);

	float3 axisp1 = frontBaseCenter - majorAxisGlobal*lSemiMajorAxisGlobal;
	float3 axisp2 = frontBaseCenter - minorAxisGlobal/focusRatio*lSemiMajorAxisGlobal;
	float2 axisp1_screen = Object2Screen(make_float4(axisp1, 1.0f), mv, pj, winW, winH);
	float2 axisp2_screen = Object2Screen(make_float4(axisp2, 1.0f), mv, pj, winW, winH);

	float thr = min(length(axisp1_screen - center) / 2, length(axisp2_screen - center) / 2);
	return dis < thr;
//	return dis < eps_pixel;
}


bool LineLens3D::PointOnLensCenter3D(float3 pos, float* mv, float* pj, int winW, int winH) {
	
	float disMajor = abs(dot(pos - c, majorAxisGlobal));
	float disMinor = abs(dot(pos - c, minorAxisGlobal));

	return disMajor < lSemiMajorAxisGlobal / 2 && disMinor < lSemiMinorAxisGlobal / focusRatio / 2;
}

bool LineLens3D::PointInCuboidRegion3D(float3 pos, float* mv, float* pj, int winW, int winH)
{
	float3 dir = pos - c;
	float dx = dot(dir, majorAxisGlobal);
	float dy = dot(dir, minorAxisGlobal);
	float dz = dot(dir, lensDir);
	return abs(dx) < lSemiMajorAxisGlobal && abs(dy)<lSemiMinorAxisGlobal / focusRatio
		&& dz>0 && dz < length(frontBaseCenter-c);
}


bool LineLens3D::PointOnOuterBoundaryWallMajorSide3D(float3 pos, float* mv, float* pj, int winW, int winH)
{
	float outerThr = 0.1;
	float upperHeight = length(frontBaseCenter - c);
	{
		float3 axisp1 = c - majorAxisGlobal*lSemiMajorAxisGlobal; //should be same with ctrlPoint3D1
		float3 dir = pos - axisp1;
		float prjMajor = dot(dir, majorAxisGlobal);
		float prjMinor = dot(dir, minorAxisGlobal);
		float prjZ = dot(dir, lensDir);
		if (prjMajor<lSemiMajorAxisGlobal*outerThr && prjMajor>-lSemiMajorAxisGlobal*0.5
			&& abs(prjMinor) < lSemiMinorAxisGlobal / focusRatio*0.5
			&& prjZ<upperHeight*(1 + outerThr) && prjZ>-upperHeight)
			return true;
	}
	{
		float3 axisp2 = c + majorAxisGlobal*lSemiMajorAxisGlobal; //should be same with ctrlPoint3D2
		float3 dir = pos - axisp2;
		float prjMajor = dot(dir, majorAxisGlobal);
		float prjMinor = dot(dir, minorAxisGlobal);
		float prjZ = dot(dir, lensDir);
		if (prjMajor<lSemiMajorAxisGlobal*outerThr && prjMajor>-lSemiMajorAxisGlobal*0.5
			&& abs(prjMinor) < lSemiMinorAxisGlobal / focusRatio*0.5
			&& prjZ<upperHeight*(1 + outerThr) && prjZ>-upperHeight)
			return true;
	}
	return false;
}

bool LineLens3D::PointOnOuterBoundaryWallMinorSide3D(float3 pos, float* mv, float* pj, int winW, int winH)
{
	float outerThr = 0.1;
	float upperHeight = length(frontBaseCenter - c);
	{
		float3 axisp1 = c - minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio;
		float3 dir = pos - axisp1;
		float prjMajor = dot(dir, majorAxisGlobal);
		float prjMinor = dot(dir, minorAxisGlobal);
		float prjZ = dot(dir, lensDir);
		if (prjMinor<lSemiMinorAxisGlobal / focusRatio*outerThr
			&& prjMinor>-lSemiMinorAxisGlobal / focusRatio*0.5
			&& abs(prjMajor) < lSemiMajorAxisGlobal*0.5
			&& prjZ<upperHeight*(1 + outerThr) && prjZ>-upperHeight)
			return true;
	}
	{
		float3 axisp2 = c + minorAxisGlobal*lSemiMinorAxisGlobal / focusRatio;
		float3 dir = pos - axisp2;
		float prjMajor = dot(dir, majorAxisGlobal);
		float prjMinor = dot(dir, minorAxisGlobal);
		float prjZ = dot(dir, lensDir);
		if (prjMinor<lSemiMinorAxisGlobal / focusRatio*outerThr
			&& prjMinor>-lSemiMinorAxisGlobal / focusRatio*0.5
			&& abs(prjMajor) < lSemiMajorAxisGlobal*0.5
			&& prjZ<upperHeight*(1 + outerThr) && prjZ>-upperHeight)
			return true;
	}
	return false;
}


void LineLens3D::ChangeClipDepth(int v, float* mv, float* pj)
{
	float speed = fmax(0.001, 0.0001*length(frontBaseCenter - estMeshBottomCenter));
	SetCenter(c - lensDir*speed*v);
}




//////////////// curve lens //////////////////////////////////


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
			if (abs(length(BezierPoints[i] - posOffsetBezierPoints[round(i *1.0 / (oriN - 1)*(posOBPsize - 1))]) - outerWidth) < outerWidth*offsetDisThr){
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
			if (abs(length(BezierPoints[i] - negOffsetBezierPoints[round(i *1.0 / (oriN - 1)*(negOBPsize - 1))]) - outerWidth) < outerWidth*offsetDisThr){
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