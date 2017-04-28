#include "LensTouchInteractor.h"
#include "DeformGLWidget.h"
#include "glwidget.h"
#include "Lens.h"


//////////////////////////////////	//for touch screen. not tested /////////////////////

void LensTouchInteractor::PinchScaleFactorChanged(float x, float y, float totalScaleFactor)
{
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	//int2 winSize = actor->GetWindowSize();
	//pickedLens = -1;
	//for (int i = 0; i < lenses->size(); i++) {
	//	Lens* l = (*lenses)[i];
	//	if (l->PointInsideInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
	//		pickedLens = i;
	//		break;
	//	}
	//}
	//if (pickedLens > -1){
	if (lenses->size()<1)
		return;
	if (INTERACT_MODE::MODIFY_LENS_DEPTH == actor->GetInteractMode()){
		//actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_DEPTH);
		//std::cout << "totalScaleFactor:" << totalScaleFactor << std::endl;
		float scaleFactor = totalScaleFactor > 1 ? 1 : -1;
		lenses->back()->ChangeClipDepth(scaleFactor, modelview, projection);
		lenses->back()->justChanged = true;
		actor->UpdateGL();
	}
	//}
	//else {
	//	actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
	//}
}

void LensTouchInteractor::ChangeLensDepth(float v)
{
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	//float scaleFactor = totalScaleFactor > 1 ? 1 : -1;
	if (lenses->size() == 0) return;
	lenses->back()->ChangeClipDepth(v, modelview, projection);

	//lenses->back()->justChanged = true;
	actor->UpdateGL();
}

bool LensTouchInteractor::InsideALens(int x, int y)
{
	if (lenses->size() < 1)
		return false;

	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	bool ret = false;
	Lens* l = lenses->back();
	if (l->PointInsideOuterBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
		ret = true;
	}
	return ret;
}

void LensTouchInteractor::UpdateLensTwoFingers(int2 p1, int2 p2)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	if (lenses->size() > 0) {
		lenses->back()->ChangeLensTwoFingers(p1, p2, modelview, projection, winSize.x, winSize.y);
		lenses->back()->justChanged = true;
		actor->UpdateGL();
	}
}

bool LensTouchInteractor::TwoPointsInsideALens(int2 p1, int2 p2)
{
	if (lenses->size() < 1)
		return false;
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	bool ret = false;
	Lens* l = lenses->back();
	if (l->PointInsideInnerBoundary(p1.x, p1.y, modelview, projection, winSize.x, winSize.y)
		&& l->PointInsideInnerBoundary(p2.x, p2.y, modelview, projection, winSize.x, winSize.y)) {
		ret = true;
	}
	return ret;
}

bool LensTouchInteractor::OnLensInnerBoundary(int2 p1, int2 p2)
{
	if (lenses->size() < 1)
		return false;
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	bool ret = false;
	Lens* l = lenses->back();
	if (l->PointOnInnerBoundary(p1.x, p1.y, modelview, projection, winSize.x, winSize.y)
		&& l->PointOnInnerBoundary(p2.x, p2.y, modelview, projection, winSize.x, winSize.y)) {
		ret = true;
	}
	return ret;
}

