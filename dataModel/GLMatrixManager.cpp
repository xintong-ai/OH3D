#include <GLMatrixManager.h>


#include <qmatrix4x4.h>
#include <helper_math.h>
#include <fstream>
#include <iostream>

#include "TransformFunc.h"

GLMatrixManager::GLMatrixManager(float3 posMin, float3 posMax)
{
	dataMin = posMin;
	dataMax = posMax;

	float3 dataCen = (dataMin + dataMax) * 0.5;
	transVec = -float3ToQvec3(dataCen);
	rotMat.setToIdentity();
	UpdateModelMatrixFromDetail();

	float3 dataRange = posMax - posMin;
	float initEyePos = min(max(max(dataRange.x, dataRange.y), dataRange.z) * 3, max(max(dataRange.x, dataRange.y), dataRange.z) + 100);	//float initEyePos = 300;

	eyeInWorld = QVector3D(0, 0, initEyePos); //may need adjust wisely. for data whose range is small, 300 is a number which is too big
	//eyeInWorld = QVector3D(0, 0, 30);
	upVecInWorld = QVector3D(0, 1, 0);
	viewVecInWorld = QVector3D(0, 0, -1);
	UpdateViewMatrixFromDetail();

	projAngle = 30;
	zNear = 0.1;
	zFar = 1000;
	UpdateProjMatrixFromDetail();

}

void GLMatrixManager::setDefaultForImmersiveMode()
{
	//give some inital values for immersive mode

	//while we have set our own mvp matrices in regular mode, the vr device can also provided _view and _projection matrix. We should apply the matrices provided by the device to achieve better stereo vision. 

	//for immersive mode, these should not be changed, since the view matrix should be close to what the device gives. but these also should not be precisely relied on, because we cannot precisely predict the viewmat received from the device
	viewVecInWorld = QVector3D(0.0f, 0.0f, -1.0f); //corresponding to the identity view matrix
	upVecInWorld = QVector3D(0.0f, 1.0f, 0.0f); //corresponding to the identity view matrix
	eyeInWorld = QVector3D(0.0f, 0.0f, 0.0f);
	viewMat.setToIdentity(); //identity view matrix is close to what VR device provided. it is equivalent to seeing from (0,0,0) to the direction of (0,0,-1), with up (0,1,0)
	
	float3 dataRange = dataMax - dataMin;
	transVec = -QVector3D(dataRange.x / 2, dataRange.y / 2, dataRange.z / 2);

	//	transVec = -QVector3D(64, 109, 107);//for 181

	rotMat.setToIdentity();			
	rotMat.rotate(-90, QVector3D(1,0,0));  //this rotation makes (0,0,1) in local overlap with the upvecinWorld
	UpdateModelMatrixFromDetail();

	projAngle = 96.73;
	zNear = 0.1;
	zFar = 1000;
}

void GLMatrixManager::setWinSize(float w, float h)
{
	width = w; height = h;
	UpdateProjMatrixFromDetail();
}

void GLMatrixManager::GetProjection(float ret[16])
{
	QMatrix4x4 pm = projMat.transposed();
	pm.copyDataTo(ret);
	return;
}

void GLMatrixManager::Scale(float v)
{
	scaleEff *= exp(v * -0.001);
	UpdateModelMatrixFromDetail();
}

void GLMatrixManager::UpdateModelMatrixFromDetail()
{
	modeMat.setToIdentity();
	modeMat = modeMat * rotMat;
	modeMat.scale(scaleEff);
	modeMat.translate(transVec);
	//justChanged = true;
	updateDepthRange();
}

void GLMatrixManager::UpdateViewMatrixFromDetail()
{
	viewMat.setToIdentity();
	viewMat.lookAt(eyeInWorld, eyeInWorld + viewVecInWorld, upVecInWorld);
	//justChanged = true;
	updateDepthRange();
}

void GLMatrixManager::UpdateProjMatrixFromDetail(float width, float height)
{
	projMat.setToIdentity();
	projMat.perspective(projAngle, (float)width / height, zNear, zFar);
	updateDepthRange();
}

void GLMatrixManager::UpdateProjMatrixFromDetail()
{
	projMat.setToIdentity();
	projMat.perspective(projAngle, (float)width / height, zNear, zFar);
}

void GLMatrixManager::GetModelViewMatrix(float mv[16])
{
	QMatrix4x4 pm = (viewMat*modeMat).transposed();
	pm.copyDataTo(mv);
	return;
}

float3 GLMatrixManager::DataCenter()
{
	return (dataMin + dataMax) * 0.5;
}

void GLMatrixManager::GetClipDepthRangeOfVol(float2 & dr)
{
	dr = clipDepthRangeOfVol;
}

void GLMatrixManager::SaveState(const char* filename)
{
	std::ofstream myfile;
	myfile.open(filename);
	for (int i = 0; i < 16; i++){
		myfile << rotMat.constData()[i] << " ";
	}
	myfile << std::endl;

	myfile << scaleEff << std::endl;

	for (int i = 0; i < 3; i++) {
		myfile << transVec[i] << " ";
	}
	myfile << std::endl;

	for (int i = 0; i < 3; i++) {
		myfile << eyeInWorld[i] << " ";
	}
	myfile << std::endl;

	for (int i = 0; i < 3; i++) {
		myfile << viewVecInWorld[i] << " ";
	}
	myfile << std::endl;

	for (int i = 0; i < 3; i++) {
		myfile << upVecInWorld[i] << " ";

	}
	myfile << std::endl;

	myfile.close();
}

void GLMatrixManager::LoadState(const char* filename)
{
	std::ifstream ifs(filename, std::ifstream::in);

	for (int i = 0; i < 16; i++) {
		ifs >> rotMat.data()[i];
	}

	ifs >> scaleEff;

	ifs >> transVec[0];
	ifs >> transVec[1];
	ifs >> transVec[2];


	ifs >> eyeInWorld[0];
	ifs >> eyeInWorld[1];
	ifs >> eyeInWorld[2];


	ifs >> viewVecInWorld[0];
	ifs >> viewVecInWorld[1];
	ifs >> viewVecInWorld[2];


	ifs >> upVecInWorld[0];
	ifs >> upVecInWorld[1];
	ifs >> upVecInWorld[2];

	UpdateModelMatrixFromDetail();
	UpdateViewMatrixFromDetail();
	UpdateProjMatrixFromDetail();
	updateDepthRange();

	ifs.close();

}
void GLMatrixManager::updateDepthRange()
{
	float modelview[16];
	float projection[16];
	GetModelViewMatrix(modelview);
	GetProjection(projection);

	float4 p[8];
	p[0] = make_float4(dataMin.x, dataMin.y, dataMin.z, 1.0f);
	p[1] = make_float4(dataMin.x, dataMin.y, dataMax.z, 1.0f);
	p[2] = make_float4(dataMin.x, dataMax.y, dataMin.z, 1.0f);
	p[3] = make_float4(dataMin.x, dataMax.y, dataMax.z, 1.0f);
	p[4] = make_float4(dataMax.x, dataMin.y, dataMin.z, 1.0f);
	p[5] = make_float4(dataMax.x, dataMin.y, dataMax.z, 1.0f);
	p[6] = make_float4(dataMax.x, dataMax.y, dataMin.z, 1.0f);
	p[7] = make_float4(dataMax.x, dataMax.y, dataMax.z, 1.0f);

	float4 pClip[8];
	std::vector<float> clipDepths;
	for (int i = 0; i < 8; i++) {
		pClip[i] = Object2Clip(p[i], modelview, projection);
		clipDepths.push_back(pClip[i].z);
	}
	clipDepthRangeOfVol.x = clamp(*std::min_element(clipDepths.begin(), clipDepths.end()), 0.0f, 1.0f);
	clipDepthRangeOfVol.y = clamp(*std::max_element(clipDepths.begin(), clipDepths.end()), 0.0f, 1.0f);
	//std::cout << "depthRange: " << depthRange.x << "," << depthRange.y << std::endl;
}