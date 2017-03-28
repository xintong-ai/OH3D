#include "MatrixLeapInteractor.h"
//#include "DeformGLWidget.h"
#include "glwidget.h"
#include "GLMatrixManager.h"
#include <iostream>

bool MatrixLeapInteractor::SlotLeftHandChanged(float3 thumpLeap, float3 indexLeap, float3 middleLeap, float3 ringLeap)
{
	bool ret = false;

#ifdef USE_OSVR
	float enterPichThr = 35, leavePinchThr = enterPichThr + 10;
#else
	float enterPichThr = 25, leavePinchThr = enterPichThr + 10; //different threshold to avoid shaking
#endif

#ifdef USE_OSVR
	float d2 = length(middleLeap - indexLeap) / 1.1;
	float d3 = length(middleLeap - ringLeap) / 1.1;
#else
	float d2 = length(middleLeap - indexLeap) / 1.2;
	float d3 = length(middleLeap - ringLeap) / 1.2;
#endif
	
	float3 curPos = GetNormalizedLeapPos((middleLeap + indexLeap + ringLeap)/3);

	switch (actor->GetInteractMode())
	{	
	case INTERACT_MODE::TRANSFORMATION:
	{
		if (d2<enterPichThr && d3<enterPichThr){
			actor->SetInteractMode(INTERACT_MODE::PANNING_MATRIX);
			frameCounter = 0;
			lastPos = curPos;
		}
		else{
		}
		ret = true;
		break;
	}
	case INTERACT_MODE::PANNING_MATRIX:
	{
		if (d2 > leavePinchThr || d3 > leavePinchThr){
			actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
		}
		else{

			int frameCounterThr = 0; //to increase stability, only panning once for every frameCounterThr+1 frames
			if (frameCounter < frameCounterThr)
			{
				frameCounter++;
			}
			else{
				float3 move = curPos - lastPos;
				float moveThr = 0.0005;
				if (length(move) > moveThr){
					float3 absMove = fabs(move);
					if (absMove.x > absMove.y && absMove.x > absMove.z){
						if (move.x > 0){
							moveViewHorizontally(1);
						}
						else{
							moveViewHorizontally(0);
						}
					}
					else if (absMove.y > absMove.x && absMove.y > absMove.z){
						if (move.y > 0){
							moveViewVertically(1);
						}
						else{
							moveViewVertically(0);
						}
					}
					else{
						if (move.z > 0){
							moveViewForwardBackward(0);
						}
						else{
							moveViewForwardBackward(1);
						}
					}
				}

				frameCounter = 0;
				lastPos = curPos;
			}
		}
		ret = true;
		break;
	}
	}

	return ret;
}


bool MatrixLeapInteractor::SlotRightHandChanged(float3 thumpLeap, float3 indexLeap, float3 middleLeap, float3 ringLeap, float &f)
{
	bool ret = false;

#ifdef USE_OSVR
	float enterPichThr = 35, leavePinchThr = enterPichThr + 10;
#else
	float enterPichThr = 25, leavePinchThr = enterPichThr + 10; //different threshold to avoid shaking
#endif

#ifdef USE_OSVR
	float d2 = length(middleLeap - indexLeap) / 1.1;
	float d3 = length(middleLeap - ringLeap) / 1.1;
#else
	float d2 = length(middleLeap - indexLeap) / 1.2;
	float d3 = length(middleLeap - ringLeap) / 1.2;
#endif
	
	float3 curPos = GetNormalizedLeapPos((middleLeap + indexLeap + ringLeap)/3);

	switch (actor->GetInteractMode())
	{	
	case INTERACT_MODE::TRANSFORMATION:
	{
		if (d2<enterPichThr && d3<enterPichThr){
			actor->SetInteractMode(INTERACT_MODE::PANNING_MATRIX);
			frameCounter = 0;
			lastPos = curPos;
		}
		else{
		}
		ret = true;
		break;
	}
	case INTERACT_MODE::PANNING_MATRIX:
	{
		if (d2 > leavePinchThr || d3 > leavePinchThr){
			actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
		}
		else{

			int frameCounterThr = 0; //to increase stability, only panning once for every frameCounterThr+1 frames
			if (frameCounter < frameCounterThr)
			{
				frameCounter++;
			}
			else{
				float3 move = curPos - lastPos;
				float moveThr = 0.0005;
				if (length(move) > moveThr){
					float rotationDegree = 2;
					if (move.x > 0){
						QMatrix4x4 oriRotMat;
						matrixMgr->GetRotMatrix(oriRotMat);
						QVector3D axis = QVector3D(oriRotMat*QVector4D(targetUpVecInLocal.x, targetUpVecInLocal.y, targetUpVecInLocal.z, 0.0));
						QMatrix4x4 newRotation;
						newRotation.rotate(-rotationDegree, axis);
						matrixMgr->setRotMat(newRotation*oriRotMat);
					}
					else{
						QMatrix4x4 oriRotMat;
						matrixMgr->GetRotMatrix(oriRotMat);
						QVector3D axis = QVector3D(oriRotMat*QVector4D(targetUpVecInLocal.x, targetUpVecInLocal.y, targetUpVecInLocal.z, 0.0));
						QMatrix4x4 newRotation;
						newRotation.rotate(rotationDegree, axis);
						matrixMgr->setRotMat(newRotation*oriRotMat);
					}
				}

				frameCounter = 0;
				lastPos = curPos;
			}
		}
		ret = true;
		break;
	}
	}

	return ret;
}

void MatrixLeapInteractor::moveViewHorizontally(int d)
{
	//d: 0. left; 1. right
	float3 horiViewInLocal = matrixMgr->getHorizontalMoveVec(targetUpVecInLocal);
	float3 newEye = matrixMgr->getEyeInLocal() + cross(horiViewInLocal, targetUpVecInLocal)*(d == 1 ? 1 : (-1));
	matrixMgr->moveEyeInLocalByModeMat(newEye);
	matrixMgr->recentChange = 3 + d;
}

void MatrixLeapInteractor::moveViewVertically(int d)
{
	//d: 0. down; 1. up
	float3 newEye = matrixMgr->getEyeInLocal() + targetUpVecInLocal*(d == 1 ? 1 : (-1));
	matrixMgr->moveEyeInLocalByModeMat(newEye);
	matrixMgr->recentChange = 5 + d;
}

void MatrixLeapInteractor::moveViewForwardBackward(int d)
{
	//d: 0. down; 1. up
	float3 newEye = matrixMgr->getEyeInLocal() + matrixMgr->getViewVecInLocal()*(d == 1 ? 1 : (-1));
	matrixMgr->moveEyeInLocalByModeMat(newEye);
	matrixMgr->recentChange = 1 + d;
}