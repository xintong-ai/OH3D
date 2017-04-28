#include "LensInteractor.h"
#include "DeformGLWidget.h"
#include "glwidget.h"
#include "Lens.h"

void LensInteractor::mousePress(int x, int y, int modifier)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	switch (actor->GetInteractMode())
	{
	case INTERACT_MODE::ADDING_LENS:
	{
		Lens* l = lenses->back();
		if (l->type == LENS_TYPE::TYPE_CURVE) {
			((CurveLens *)l)->AddCtrlPoint(x, y);
		}
		else if (l->type == LENS_TYPE::TYPE_LINE) {
			((LineLens *)l)->ctrlPointScreen1 = make_float2(x, y);
			((LineLens *)l)->ctrlPointScreen2 = make_float2(x, y);
		}
		break;
	}
	case INTERACT_MODE::TRANSFORMATION:
	{
		if (lenses->size()<1)
			return;
		Lens* l = lenses->back();
		if (l->PointOnLensCenter(x, y, modelview, projection, winSize.x, winSize.y)) {
			actor->SetInteractMode(INTERACT_MODE::MOVE_LENS);
			break;
		}
		else if (l->PointOnOuterBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
			actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE);
			break;
		}
		else if (l->PointOnInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
			actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE);
			std::cout << "hehaherha" << std::endl;
			break;
		}
		break;
	}

	}
	lastPt = make_int2(x, y);
}


void LensInteractor::mouseRelease(int x, int y, int modifier)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	float3 posMin, posMax;
	actor->GetPosRange(posMin, posMax);

	if (INTERACT_MODE::ADDING_LENS == actor->GetInteractMode()) {
		Lens* l = lenses->back();
		if (l->type == LENS_TYPE::TYPE_LINE) {
			if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE)
			{
				((LineLens*)l)->FinishConstructing(modelview, projection, winSize.x, winSize.y);
			}
			else
			{
				((LineLens3D*)l)->FinishConstructing(modelview, projection, winSize.x, winSize.y, posMin, posMax);
			}
		}
		else if (l->type == LENS_TYPE::TYPE_CURVE) {
			((CurveLens *)l)->FinishConstructing(modelview, projection, winSize.x, winSize.y);
		}
		l->justChanged = true;
		actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
	}
	else {
		if (actor->GetInteractMode() == INTERACT_MODE::MOVE_LENS || actor->GetInteractMode() == INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE || actor->GetInteractMode() == INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE){
			if (lenses->size() > 0){
				lenses->back()->justChanged = true;
			}
		}
		else if (actor->GetInteractMode() == INTERACT_MODE::TRANSFORMATION && changeLensWhenRotateData){
			//this decides whether to relocate the mesh when rotating the data
			if (lenses->size() > 0){
				Lens* l = lenses->back();
				l->justChanged = true;

				if (l->type == LENS_TYPE::TYPE_LINE && ((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE){
					((LineLens3D*)l)->UpdateObjectLineLens(winSize.x, winSize.y, modelview, projection, posMin, posMax);
				}
			}
		}

		if (actor->GetInteractMode() == INTERACT_MODE::MOVE_LENS && isSnapToFeature){
			/*// !!! DON'T DELETE !!!
			//these code will be process later
			GlyphRenderable* glyphRenderable = (GlyphRenderable*)actor->GetRenderable("glyph");
			Lens* l = (*lenses)[lenses->size() - 1];
			float3 center = make_float3(l->GetCenter());
			int resid = -1;
			if (glyphRenderable->findClosetFeature(center, snapPos, resid))
			{
			l->SetCenter(snapPos);

			PolyRenderable* r1 = (PolyRenderable*)actor->GetRenderable("ventricles");
			PolyRenderable* r2 = (PolyRenderable*)actor->GetRenderable("tumor1");
			PolyRenderable* r3 = (PolyRenderable*)actor->GetRenderable("tumor2");
			if (resid == 1){
			r1->isSnapped = true;
			r2->isSnapped = false;
			r3->isSnapped = false;
			}
			else if (resid == 2){
			r1->isSnapped = false;
			r2->isSnapped = true;
			r3->isSnapped = false;
			}
			else if (resid == 3){
			r1->isSnapped = false;
			r2->isSnapped = false;
			r3->isSnapped = true;
			}
			}
			*/
		}
	}
	actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
}

void LensInteractor::mouseMove(int x, int y, int modifier)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	float3 posMin, posMax;
	actor->GetPosRange(posMin, posMax);
	switch (actor->GetInteractMode())
	{
	case INTERACT_MODE::ADDING_LENS:
	{
		Lens* l = lenses->back();
		if (l->type == LENS_TYPE::TYPE_CURVE){
			((CurveLens *)l)->AddCtrlPoint(x, y);
		}
		else if (l->type == LENS_TYPE::TYPE_LINE){
			((LineLens *)l)->ctrlPointScreen2 = make_float2(x, y);
			if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE){
				((LineLens*)l)->UpdateInfoFromCtrlPoints(modelview, projection, winSize.x, winSize.y);
			}
		}
		break;
	}
	case INTERACT_MODE::MOVE_LENS:
	{
		float3 moveVec = lenses->back()->MoveLens(x, y, modelview, projection, winSize.x, winSize.y);
		if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE && lenses->back()->type == LENS_TYPE::TYPE_LINE){
			((LineLens3D*)lenses->back())->UpdateObjectLineLens(winSize.x, winSize.y, modelview, projection, posMin, posMax);
		}

		if (isSnapToGlyph){
			/*// !!! DON'T DELETE !!!
			//these code will be process later
			DeformGlyphRenderable* glyphRenderable = (DeformGlyphRenderable*)actor->GetRenderable("glyph");
			glyphRenderable->findClosetGlyph(make_float3((*lenses)[pickedLens]->GetCenter()));
			*/
		}
		else if (isSnapToFeature){
			/*// !!! DON'T DELETE !!!
			//these code will be process later
			GlyphRenderable* glyphRenderable = (GlyphRenderable*)actor->GetRenderable("glyph");
			snapPos;
			int resid=-1;
			glyphRenderable->findClosetFeature(make_float3((*lenses)[pickedLens]->GetCenter()), snapPos, resid);

			PolyRenderable* r1 = (PolyRenderable*)actor->GetRenderable("ventricles");
			PolyRenderable* r2 = (PolyRenderable*)actor->GetRenderable("tumor1");
			PolyRenderable* r3 = (PolyRenderable*)actor->GetRenderable("tumor2");
			if (resid == 1){
			r1->isSnapped = true;
			r2->isSnapped = false;
			r3->isSnapped = false;
			}
			else if (resid == 2){
			r1->isSnapped = false;
			r2->isSnapped = true;
			r3->isSnapped = false;
			}
			else if (resid == 3){
			r1->isSnapped = false;
			r2->isSnapped = false;
			r3->isSnapped = true;
			}
			*/
		}
		break;
	}
	case INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE:
	{
		if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE)
			lenses->back()->ChangeLensSize(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);
		else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE){
			lenses->back()->ChangeObjectLensSize(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);

			if (lenses->back()->type == TYPE_LINE){
				((LineLens3D*)(lenses->back()))->UpdateObjectLineLens(winSize.x, winSize.y, modelview, projection, posMin, posMax);
			}
		}
		break;
	}
	case INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE:
	{
		if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE)
			lenses->back()->ChangeFocusRatio(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);
		else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE)
			lenses->back()->ChangeObjectFocusRatio(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);
		break;
	}
	}
	lastPt = make_int2(x, y);
}



bool LensInteractor::MouseWheel(int x, int y, int modifier, int delta)
{
	if (lenses->size() < 1)
		return false;

	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	bool insideAnyLens = false;
	Lens* l = lenses->back();
	if (l->PointInsideInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
		insideAnyLens = true;
		float3 posMin, posMax;
		actor->GetPosRange(posMin, posMax);
		float3 dif = posMax - posMin;
		float coeff = min(min(dif.x, dif.y), dif.z) / 10.0 / 20.0 / 20.0;
		l->ChangeClipDepth(delta*coeff, modelview, projection);
	}
	return insideAnyLens;
}
