#include "ImmersiveInteractor.h"
#include "GLMatrixManager.h"
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>

void ImmersiveInteractor::mouseMoveMatrix(float fromX, float fromY, float toX, float toY, int modifier, int mouseKey)
{
	if (!noMoveMode){
		if (mouseKey == 1){
			Rotate(fromX, fromY, toX, toY);
		}
		else if (mouseKey == 2){
			Translate(toX - fromX, toY - fromY);
		}
	}
}

void ImmersiveInteractor::Rotate(float fromX, float fromY, float toX, float toY)
{
	if (!isActive)
		return;

	float3 upInLocalf3 = matrixMgr->getUpInLocal();
	QVector3D upInLocal = QVector3D(upInLocalf3.x, upInLocalf3.y, upInLocalf3.z);
	float close = abs(QVector3D::dotProduct(upInLocal, targetUpVecInLocal));

	float angthr = 0.5;
	
	// *rot = trackball->rotate(fromX, fromY, toX, toY);
	//*rot = trackball->rotate(toX, toY, fromX, fromY);  //note the from and to is different from normal
	*rot = trackball->rotate(toX, 0, fromX, 0);  //raising or lowering the head should be done by using the device
	float m[16];
	rot->matrix(m);
	QMatrix4x4 newRotation = QMatrix4x4(m).transposed();
	

	/*

	float m[16];

	*rot = trackball->rotate(toX, toY, toX, fromY);  //note the from and to is different from normal
	rot->matrix(m);
	QMatrix4x4 newRotationY = QMatrix4x4(m).transposed();

	*rot = trackball->spin(toX, fromY, fromX, fromY);  //note the from and to is different from normal
	rot->matrix(m);

	QMatrix4x4 newRotationX = QMatrix4x4(m).transposed();

	QMatrix4x4 newRotation = newRotationX*newRotationY;
	//newRotation = newRotation2;
	*/


	/*
	QMatrix4x4 newRotation1;
	newRotation1.rotate(15 * (fromX - toX), upInLocal);
	float3 sideInLocalf3 = cross(matrixMgr->getUpInLocal(), matrixMgr->getViewVecInLocal());
	QVector3D sideInLocal = QVector3D(sideInLocalf3.x, sideInLocalf3.y, sideInLocalf3.z);
	QMatrix4x4 newRotation2;
	newRotation2.rotate(15 * (fromY - toY), sideInLocal);
	newRotation = newRotation2*newRotation1;
	*/



	QMatrix4x4 oriRotMat;
	matrixMgr->GetRotMatrix(oriRotMat);

	QVector3D upVecInWorld = matrixMgr->getUpVecInWorld();
	QVector3D tempUpInLocal = (QVector3D((newRotation*oriRotMat).inverted()*QVector4D(upVecInWorld, 0.0))).normalized();

	close = abs(QVector3D::dotProduct(tempUpInLocal, targetUpVecInLocal));
	if (close > angthr){
		//matrixMgr->setRotMat(oriRotMat*newRotation);
		matrixMgr->setRotMat(newRotation*oriRotMat);
	}
};


void ImmersiveInteractor::Translate(float x, float y)
{
	return;
};


void ImmersiveInteractor::mousePress(int x, int y, int modifier, int mouseKey)
{
	if (noMoveMode){
		if (mouseKey == 1)
		{
			matrixMgr->toRotateLeft = true;
		}
		else if (mouseKey == 2)
		{
			matrixMgr->toRotateRight = true;
		}
	}
}

void ImmersiveInteractor::mouseRelease(int x, int y, int modifier)
{
	if (noMoveMode){
		matrixMgr->toRotateLeft = false;
		matrixMgr->toRotateRight = false;
	}
}


bool ImmersiveInteractor::MouseWheel(int x, int y, int modifier, float v)
{
	if (!isActive)
		return false;

	float3 viewVecLocal = matrixMgr->getViewVecInLocal();
	QVector3D oldTransVec = matrixMgr->getTransVec();
	matrixMgr->setTransVec(oldTransVec - v / 100.0 *QVector3D(viewVecLocal.x, viewVecLocal.y, viewVecLocal.z));
	if (v > 0){
		matrixMgr->recentChange = 1;
	}
	else{
		matrixMgr->recentChange = 2;
	}
}

void ImmersiveInteractor::keyPress(char key)
{
	switch (key)
	{
	case 'a':
	case 'A':
		moveViewHorizontally(0);
		break;
	case 'd':
	case 'D':
		moveViewHorizontally(1);
		break;
	case 'w':
	case 'W':
		moveViewVertically(1);
		break;
	case 's':
	case 'S':
		moveViewVertically(0);
		break;
	}
}

void ImmersiveInteractor::moveViewHorizontally(int d)
{
	//d: 0. left; 1. right
	float3 horiViewInLocal = matrixMgr->getHorizontalMoveVec(make_float3(targetUpVecInLocal.x(), targetUpVecInLocal.y(), targetUpVecInLocal.z()));
	float3 newEye = matrixMgr->getEyeInLocal() + cross(horiViewInLocal, make_float3(targetUpVecInLocal.x(), targetUpVecInLocal.y(), targetUpVecInLocal.z()))*(d==1?1:(-1));
	matrixMgr->moveEyeInLocalByModeMat(newEye);
	matrixMgr->recentChange = 3 + d;
}

void ImmersiveInteractor::moveViewVertically(int d)
{
	//d: 0. down; 1. up
	float3 newEye = matrixMgr->getEyeInLocal() + make_float3(targetUpVecInLocal.x(), targetUpVecInLocal.y(), targetUpVecInLocal.z())*(d==1?1:(-1));
	matrixMgr->moveEyeInLocalByModeMat(newEye);
	matrixMgr->recentChange = 5 + d;
}