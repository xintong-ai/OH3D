#include "ImmersiveInteractor.h"
#include "GLMatrixManager.h"
#include "mouse/trackball.h"

void ImmersiveInteractor::Rotate(float fromX, float fromY, float toX, float toY)
{
	if (!isActive)
		return;

	//suppose the rotation happens in eye coordinate
	matrixMgr->getTrackBall()->rotate(fromX, fromY, toX, toY);
	//the rotation in trackball is not strictly in eye coordinate. but use this as approximation
	QVector3D axis;
	matrixMgr->getTrackBall()->getRotationAxis(axis[0], axis[1], axis[2]);
	float a = matrixMgr->getTrackBall()->getAngleRotated();

	QMatrix4x4 m,mv, v;

	matrixMgr->GetModelMatrix(m);
	matrixMgr->GetModelViewMatrix(mv);
	matrixMgr->GetViewMatrix(v);

	
	QVector3D eyeInLocal = m.inverted().map(matrixMgr->getEyeVecInWorld());
	
	QMatrix4x4 r;
	r.rotate(a / 3.14159*180.0, axis);

	QVector3D cofInEye = mv.map(matrixMgr->getCofLocal());
	QVector3D newCofInEye = r.map(cofInEye);
	QVector3D newCofLocal = (mv.inverted()).map(newCofInEye);


	QVector3D UpVecTipInEye = v.map(matrixMgr->getEyeVecInWorld() + matrixMgr->getUpVecInWorld());
	QVector3D newUpVecTipInEye = r.map(UpVecTipInEye);
	QVector3D newUpVecTipLocal = (mv.inverted()).map(newUpVecTipInEye);
	
	matrixMgr->setCofLocal(newCofLocal);

	matrixMgr->GetModelMatrix(m);
	QVector3D newEyeInWorld = m.map(eyeInLocal);
	matrixMgr->setEyeAndUpInWorld(newEyeInWorld, m.map(newUpVecTipLocal) - newEyeInWorld); 

	return; 
};


void ImmersiveInteractor::Translate(float x, float y)
{
	if (!isActive)
		return;

	QMatrix4x4 m, mv;
	matrixMgr->GetModelMatrix(m);

	QVector3D eyeInWorld = matrixMgr->getEyeVecInWorld();
	QVector3D dir = QVector3D::crossProduct(eyeInWorld, matrixMgr->getUpVecInWorld())*x ;
		
	QVector3D newCofInWorld = dir;

	matrixMgr->setCofLocal(m.inverted().map(newCofInWorld));

	return;
};

bool ImmersiveInteractor::MouseWheel(int x, int y, int modifier, float v)
{
	if (!isActive)
		return false;

	QMatrix4x4 m, mv;
	matrixMgr->GetModelMatrix(m);

	QVector3D eyeInWorld = matrixMgr->getEyeVecInWorld();
	QVector3D dir = eyeInWorld.normalized()*v/-1000.0;
	QVector3D newCofInWorld = dir;

	matrixMgr->setCofLocal(m.inverted().map(newCofInWorld));
	
	return true;
}
