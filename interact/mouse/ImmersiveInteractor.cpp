#include "ImmersiveInteractor.h"
#include "GLMatrixManager.h"

void ImmersiveInteractor::Rotate(float fromX, float fromY, float toX, float toY)
{
	if (!isActive)
		return;
	//*rot = trackball->rotate(fromX, fromY, toX, toY);
	*rot = trackball->rotate(toX, toY, fromX, fromY);  //note the from and to is different from normal

	float m[16];
	rot->matrix(m);
	matrixMgr->applyPreRotMat(QMatrix4x4(m).transposed());
};


void ImmersiveInteractor::Translate(float x, float y)
{/*
	if (!isActive)
		return;

	QMatrix4x4 m, mv;
	matrixMgr->GetModelMatrix(m);

	QVector3D eyeInWorld = matrixMgr->getEyeVecInWorld();
	QVector3D dir = QVector3D::crossProduct(eyeInWorld, matrixMgr->getUpVecInWorld())*x ;
		
	QVector3D newCofInWorld = dir;

	matrixMgr->setCofLocal(m.inverted().map(newCofInWorld));
	*/
	return;
};

bool ImmersiveInteractor::MouseWheel(int x, int y, int modifier, float v)
{
	if (!isActive)
		return false;

	float3 viewVecLocal = matrixMgr->getViewVecInLocal();
	QVector3D oldTransVec = matrixMgr->getTransVec();
	matrixMgr->setTransVec(oldTransVec - v / 100.0 *QVector3D(viewVecLocal.x, viewVecLocal.y, viewVecLocal.z));
}
