#include "ImmersiveInteractor.h"
#include "GLMatrixManager.h"
#include "mouse/trackball.h"

void ImmersiveInteractor::Rotate(float fromX, float fromY, float toX, float toY, std::shared_ptr<GLMatrixManager> matrixMgr)
{
	matrixMgr->getTrackBall()->rotate(fromX, fromY, toX, toY);
	//the rotation in trackball is not strictly in eye coordinate. but use this as approximation
	QVector3D axis;
	matrixMgr->getTrackBall()->getRotationAxis(axis[0], axis[1], axis[2]);
	float a = matrixMgr->getTrackBall()->getAngleRotated();

	QMatrix4x4 m,mv;

	matrixMgr->GetModelMatrix(m);
	matrixMgr->GetModelViewMatrix(mv);
	
	
	QVector3D eyeInLocal = m.inverted().map(matrixMgr->getEyeVecInWorld());
	


	QVector3D cofInEye = mv.map(matrixMgr->getCofLocal());
	QMatrix4x4 r;
	r.rotate(a/3.14159*180.0, axis);
	QVector3D newCofInEye = r.map(cofInEye);
	QVector3D newCofLocal = (mv.inverted()).map(newCofInEye);

	matrixMgr->setCofLocal(newCofLocal);

	matrixMgr->GetModelMatrix(m);
	matrixMgr->setEyeInWorld(m.map(eyeInLocal));

	return; 
};


void ImmersiveInteractor::Translate(float x, float y, std::shared_ptr<GLMatrixManager> matrixMgr)
{
	QMatrix4x4 m, mv;
	matrixMgr->GetModelMatrix(m);

	QVector3D eyeInWorld = matrixMgr->getEyeVecInWorld();
	QVector3D dir = QVector3D::crossProduct(eyeInWorld, matrixMgr->getUpVecInWorld())*x ;
		
	QVector3D newCofInWorld = dir;

	matrixMgr->setCofLocal(m.inverted().map(newCofInWorld));

	return;
};

void ImmersiveInteractor::wheelEvent(float v, std::shared_ptr<GLMatrixManager> matrixMgr)
{
	QMatrix4x4 m, mv;
	matrixMgr->GetModelMatrix(m);

	QVector3D eyeInWorld = matrixMgr->getEyeVecInWorld();
	QVector3D dir = eyeInWorld.normalized()*v/-1000.0;
	QVector3D newCofInWorld = dir;

	matrixMgr->setCofLocal(m.inverted().map(newCofInWorld));
	
	return;
}
