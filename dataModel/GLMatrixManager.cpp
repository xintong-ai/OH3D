#include <GLMatrixManager.h>


#include <qmatrix4x4.h>
#include <helper_math.h>
#include <fstream>
#include <iostream>


GLMatrixManager::GLMatrixManager(float3 posMin, float3 posMax)
{
	dataMin = posMin;
	dataMax = posMax;

	float3 dataCen = (dataMin + dataMax) * 0.5;
	transVec = -float3ToQvec3(dataCen);
	rotMat.setToIdentity();
	UpdateModelMatrixFromDetail();

	eyeInWorld = QVector3D(0, 0, 300); //may need adjust wisely. for data whose range is small, 300 is a number which is too big
	//eyeInWorld = QVector3D(0, 0, 30);
	upVecInWorld = QVector3D(0, 1, 0);
	viewVecInWorld = QVector3D(0, 0, -1);
	UpdateViewMatrixFromDetail();

	projAngle = 30;
	zNear = 0.1;
	zFar = 1000;
}

void GLMatrixManager::setDefaultForImmersiveMode()  //must be called after setVol()
{
	//give some inital values for immersive mode

	//while we have set our own mvp matrices in regular mode, the vr device can also provided _view and _projection matrix. We should apply the matrices provided by the device to achieve better stereo vision. 

	//for immersive mode, these should not be changed, since the view matrix should be close to what the device gives. but these also should not be precisely relied on, because we cannot precisely predict the viewmat received from the device
	viewVecInWorld = QVector3D(0.0f, 0.0f, -1.0f); //corresponding to the identity view matrix
	upVecInWorld = QVector3D(0.0f, 1.0f, 0.0f); //corresponding to the identity view matrix
	eyeInWorld = QVector3D(0.0f, 0.0f, 0.0f);
	viewMat.setToIdentity(); //identity view matrix is close to what VR device provided. it is equivalent to seeing from (0,0,0) to the direction of (0,0,-1), with up (0,1,0)
	
	transVec = -QVector3D(64, 109, 107);
	rotMat.setToIdentity();			
	rotMat.rotate(-90, QVector3D(1,0,0));  //this rotation makes (0,0,1) in local overlap with the upvecinWorld
	UpdateModelMatrixFromDetail();

	projAngle = 96.73;
	zNear = 0.1;
	zFar = 1000;
}


void GLMatrixManager::GetProjection(float ret[16], float width, float height)
{
	UpdateProjMatrixFromDetail(width, height);//different from mv matrix, prj relys on width and height
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
	justChanged = true;
}

void GLMatrixManager::UpdateViewMatrixFromDetail()
{
	viewMat.setToIdentity();
	viewMat.lookAt(eyeInWorld, eyeInWorld + viewVecInWorld, upVecInWorld);
	justChanged = true;
}

void GLMatrixManager::UpdateProjMatrixFromDetail(float width, float height)
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

void GLMatrixManager::SaveState(const char* filename)
{
	std::ofstream myfile;
	myfile.open(filename);
	for (int i = 0; i < 16; i++){
		myfile << rotMat.constData()[i] << " ";
	}
	myfile << std::endl;

	for (int i = 0; i < 3; i++) {
		myfile << transVec[i] << " ";
	}
	myfile << std::endl;

	myfile << scaleEff << std::endl;
	myfile.close();
}

void GLMatrixManager::LoadState(const char* filename)
{
	std::ifstream ifs(filename, std::ifstream::in);

	for (int i = 0; i < 16; i++) {
		ifs >> rotMat.data()[i];
	}

	ifs >> transVec[0];
	ifs >> transVec[1];
	ifs >> transVec[2];

	ifs >> scaleEff;

	ifs.close();

}
