#include <iostream>
#include <helper_math.h>
#include "AnimationByMatrixProcessor.h"
#include "GLMatrixManager.h"

void AnimationByMatrixProcessor::startAnimation()
{
	if (views.size() > 1){
		animeStarted = true;
		start = std::clock();
	}
	else{
		std::cout << "views not set correctly! animation cannot start" << std::endl;
	}
}

bool AnimationByMatrixProcessor::process(float modelview[16], float projection[16], int winW, int winH)
{
	if (!isActive)
		return false;

	//theoretically the following belongs to the job of interactor. however qt interactor does not well support mouse key pressint-and-holding, so put it here
	float rotationDegree = 2;
	if (matrixMgr->toRotateLeft){
		QMatrix4x4 oriRotMat;
		matrixMgr->GetRotMatrix(oriRotMat);
		QVector3D axis = QVector3D(oriRotMat*QVector4D(targetUpVecInLocal.x, targetUpVecInLocal.y, targetUpVecInLocal.z, 0.0));
		QMatrix4x4 newRotation;
		newRotation.rotate(-rotationDegree, axis);
		matrixMgr->setRotMat(newRotation*oriRotMat);
	}
	else if (matrixMgr->toRotateRight){
		QMatrix4x4 oriRotMat;
		matrixMgr->GetRotMatrix(oriRotMat);
		QVector3D axis = QVector3D(oriRotMat*QVector4D(targetUpVecInLocal.x, targetUpVecInLocal.y, targetUpVecInLocal.z, 0.0));
		QMatrix4x4 newRotation;
		newRotation.rotate(rotationDegree, axis);
		matrixMgr->setRotMat(newRotation*oriRotMat);
	}
	else if (animeStarted){	
		double past = (std::clock() - start) / (double)CLOCKS_PER_SEC;
		if (past > totalDuration){
			//end animation
			isActive = false;
			std::cout << "animiation ends" << std::endl;
			return false;
		}

		int n = views.size();
		double p = past / totalDuration * (n - 1);
		int n1 = floor(p), n2 = n1 + 1;
		if (n2 < n){
			float3 view = views[n1] * (n2 - p) + views[n2] * (p - n1);
			//std::cout << view.x << " " << view.y << " " << view.z << std::endl;
			matrixMgr->moveEyeInLocalByModeMat(view);
		}
		return true;
	}
	else{
		return false;
	}
}

