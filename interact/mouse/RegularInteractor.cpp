#include "RegularInteractor.h"
#include "GLMatrixManager.h"
#include "mouse/trackball.h"
#include "Rotation.h"

void RegularInteractor::Rotate(float fromX, float fromY, float toX, float toY)
{
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
	matrixMgr->TranslateInWorldSpace(x, y);
	return; 
};

void RegularInteractor::wheelEvent(float v)
{
	matrixMgr->Scale(v);
	return;
}
