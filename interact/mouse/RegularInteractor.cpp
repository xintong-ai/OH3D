#include "RegularInteractor.h"
#include "GLMatrixManager.h"
#include "mouse/trackball.h"
#include "Rotation.h"

void RegularInteractor::Rotate(float fromX, float fromY, float toX, float toY)
{
	if (!isActive)
		return; 
	Rotation *rot;
	rot = new Rotation();
	*rot = matrixMgr->getTrackBall()->rotate(fromX, fromY, toX, toY);
	float m[16];
	rot->matrix(m);
	matrixMgr->applyPreRotMat(QMatrix4x4(m).transposed());
	delete rot;
	return;
};

void RegularInteractor::Translate(float x, float y)
{
	if (!isActive)
		return;
	matrixMgr->TranslateInWorldSpace(x, y);
	return; 
};

bool RegularInteractor::MouseWheel(int x, int y, int modifier, float v)
{
	if (!isActive)
		return false;
	matrixMgr->Scale(v);
	return true;
}
