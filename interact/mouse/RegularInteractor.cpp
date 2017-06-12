#include "RegularInteractor.h"
#include "GLMatrixManager.h"

void RegularInteractor::mouseMoveMatrix(float fromX, float fromY, float toX, float toY, int modifier, int mouseKey)
{
	if (mouseKey == 1){
		Rotate(fromX, fromY, toX, toY);
	}
	else if (mouseKey == 2){
		Translate(toX - fromX, toY - fromY);
	}
}

void RegularInteractor::Rotate(float fromX, float fromY, float toX, float toY)
{
	if (!isActive)
		return; 
	*rot = trackball->rotate(fromX, fromY, toX, toY);
	float m[16];
	rot->matrix(m);
	matrixMgr->applyPreRotMat(QMatrix4x4(m).transposed());
	return;
};

void RegularInteractor::Translate(float x, float y)
{
	if (!isActive)
		return;

	float scale = 10;

	QVector3D eyeInWorld = matrixMgr->getEyeVecInWorld();
	QVector3D upVecInWorld = matrixMgr->getUpVecInWorld();
	QVector3D viewVecWorld = matrixMgr->getViewVecInWorld();
	QVector3D newEyeInWorld = eyeInWorld + x*scale* (QVector3D::crossProduct(upVecInWorld, viewVecWorld)).normalized() - y*scale*(upVecInWorld.normalized());
	matrixMgr->setEyeInWorld(newEyeInWorld);
	return; 
};

bool RegularInteractor::MouseWheel(int x, int y, int modifier, float v)
{
	if (!isActive)
		return false;
	
	QVector3D eyeInWorld = matrixMgr->getEyeVecInWorld();
	QVector3D viewVecWorld = matrixMgr->getViewVecInWorld();
	matrixMgr->setEyeInWorld(eyeInWorld + v*viewVecWorld * matrixMgr->GetVolScale()*0.01);

	return true;
}
