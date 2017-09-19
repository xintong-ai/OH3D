#include "ImmersiveInteractor.h"
#include "GLMatrixManager.h"
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>

void ImmersiveInteractor::mouseMoveMatrix(float fromX, float fromY, float toX, float toY, int modifier, int mouseKey)
{
	if (!noMoveMode){
		if (mouseKey == 1){
			RotateLocal(fromX, fromY, toX, toY);
		}
		else if (mouseKey == 2){
			RotateEye(fromX, fromY, toX, toY);
		}
	}
}

void ImmersiveInteractor::RotateLocal(float fromX, float fromY, float toX, float toY)
{
	if (!isActive)
		return;

	float3 upInLocalf3 = matrixMgr->getUpInLocal();
	QVector3D upInLocal = QVector3D(upInLocalf3.x, upInLocalf3.y, upInLocalf3.z);
	float close = abs(QVector3D::dotProduct(upInLocal, targetUpVecInLocal));

	float angthr = 0.5;

	*rot = trackball->rotate(toX, 0, fromX, 0);  //raising or lowering the head should be done by using the device
	float m[16];
	rot->matrix(m);
	QMatrix4x4 newRotation = QMatrix4x4(m).transposed();
	
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

void ImmersiveInteractor::RotateEye(float fromX, float fromY, float toX, float toY)
{
	QVector3D eyeInWorld = matrixMgr->getEyeVecInWorld();
	QVector3D upVecInWorld = matrixMgr->getUpVecInWorld();
	QVector3D viewVecInWorld = matrixMgr->getViewVecInWorld();

	*rot = trackball->rotate(0, fromY, 0, toY);  //left or right the head has been done by RotateLocal()
	float m[16];
	rot->matrix(m);
	QMatrix4x4 newRotation = QMatrix4x4(m).transposed();

	matrixMgr->setViewAndUpInWorld(QVector3D(newRotation*QVector4D(viewVecInWorld, 0)), QVector3D(newRotation*QVector4D(upVecInWorld, 0)));
}


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
	matrixMgr->setTransVec(oldTransVec - v / 400.0 *QVector3D(viewVecLocal.x, viewVecLocal.y, viewVecLocal.z));
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
		/*case 'z':
	case 'Z':
		if (ve != 0 && infoGuideRenderable != 0){
			//for Tao09Detail
			ve->currentMethod = VPMethod::Tao09Detail;
			ve->compute_SkelSampling(VPMethod::Tao09Detail);
			infoGuideRenderable->changeWhetherGlobalGuideMode(true);

			//for LabelVisibility from file
			//if (!hasLabelFromFile){
			//	labelVolCUDA->VolumeCUDA_contentUpdate(labelVolLocal, 1, 1);
			//	std::cout << std::endl << "The lable volume has been updated from drawing" << std::endl << std::endl;
			//}
			ve->currentMethod = VPMethod::LabelVisibility;
			ve->compute_SkelSampling(VPMethod::LabelVisibility);
			infoGuideRenderable->changeWhetherGlobalGuideMode(true);
		}
		else{
			std::cout << "ve or infoGuideRenderable not set!!" << std::endl << std::endl;
		}
		break;
	case 'x':
	case 'X':
		if (infoGuideRenderable != 0){
			infoGuideRenderable->changeWhetherGlobalGuideMode(false);
		}
		else{
			std::cout << "ve or infoGuideRenderable not set!!" << std::endl << std::endl;
		}
		break;*/
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