#include "LensLeapInteractor.h"
#include "DeformGLWidget.h"
#include "glwidget.h"
#include "Lens.h"
#include "Particle.h"

bool LensLeapInteractor::SlotOneHandChanged(float3 thumpLeap, float3 indexLeap, float3 middleLeap, float3 ringLeap,  float &f)

{
	//currently only work for line lens 3D

	//std::cout << "thumpLeap " << thumpLeap.x << " " << thumpLeap.y << " " << thumpLeap.z << std::endl;
	//std::cout << "indexLeap " << indexLeap.x << " " << indexLeap.y << " " << indexLeap.z << std::endl;

	float4 markerPos;
	float valRight;

	bool ret;

#ifdef USE_OSVR
	float enterPichThr = 35, leavePinchThr = enterPichThr + 10;
#else
	float enterPichThr = 25, leavePinchThr = enterPichThr + 10; //different threshold to avoid shaking
#endif

	float d = length(thumpLeap - indexLeap);

	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);

	float3 curPos = GetTransferredLeapPos(indexLeap);
	//float3 curPos = (GetTransferredLeapPos(thumpLeap) + GetTransferredLeapPos(indexLeap)) / 2;


	markerPos = make_float4(curPos, 1.0);


	if (lenses->size() == 0){
		float3 posMin, posMax;
		actor->GetPosRange(posMin, posMax);
		if (actor->GetInteractMode() == INTERACT_MODE::TRANSFORMATION && d < enterPichThr && !outOfDomain(curPos, posMin, posMax)){

			Lens* l = new LineLens3D(actor->DataCenter(), 0.3);
			l->isConstructedFromLeap = true;

			lenses->push_back(l);
			//l->justChanged = true; //constructing first, then set justChanged
			actor->UpdateGL();
			actor->SetInteractMode(INTERACT_MODE::ADDING_LENS);
			((LineLens3D *)l)->ctrlPoint3D1 = curPos;
			((LineLens3D *)l)->ctrlPoint3D2 = curPos;
			valRight = 1;
			ret = true;
		}
		else{
			//add global rotation?

			if (outOfDomain(curPos, posMin, posMax)){
				ret = false;
			}
			else{
				ret = true;
			}
			ret = true;
		}
	}
	else{	//lenses->size()>0
		LineLens3D* l = (LineLens3D*)lenses->back();


#ifdef USE_OSVR
		float d2 = length(middleLeap - indexLeap) / 1.1;
		float d3 = length(middleLeap - ringLeap) / 1.1;
#else
		float d2 = length(middleLeap - indexLeap) / 1.2;
		float d3 = length(middleLeap - ringLeap) / 1.2;
#endif



		switch (actor->GetInteractMode())
		{
		case INTERACT_MODE::ADDING_LENS:
		{
			if (d < leavePinchThr){
				l->ctrlPoint3D2 = curPos;
				ret = true;
			}
			else {
				float3 posMin, posMax;
				actor->GetPosRange(posMin, posMax);
				l->FinishConstructing3D(modelview, projection, winSize.x, winSize.y, posMin, posMax);

				l->justChanged = true;

				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				ret = true;
			}
			break;
		}
		case INTERACT_MODE::TRANSFORMATION:
		{
		if (d < enterPichThr){
			valRight = 1;
			if (l->PointOnLensCenter3D(curPos, modelview, projection, winSize.x, winSize.y)){
				actor->SetInteractMode(INTERACT_MODE::MOVE_LENS);
				prevPos = curPos;
				prevPointOfLens = l->c;
				std::cout << "haha" << std::endl;
			}
			else if (l->PointOnOuterBoundaryWallMajorSide3D(curPos, modelview, projection, winSize.x, winSize.y)){
				actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE);
				prevPos = curPos;
				if (length(curPos - l->ctrlPoint3D1) < length(curPos - l->ctrlPoint3D2)){
					prevPointOfLens = l->ctrlPoint3D1;
				}
				else{
					prevPointOfLens = l->ctrlPoint3D2;
				}
			}
			else if (l->PointOnOuterBoundaryWallMinorSide3D(curPos, modelview, projection, winSize.x, winSize.y)){
				actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE);
				prevPos = curPos;
				float3 pp1 = l->c - l->minorAxisGlobal*l->lSemiMinorAxisGlobal / l->focusRatio;
				float3 pp2 = l->c + l->minorAxisGlobal*l->lSemiMinorAxisGlobal / l->focusRatio;
				if (length(curPos - pp1) < length(curPos - pp2)){
					prevPointOfLens = pp1;
				}
				else{
					prevPointOfLens = pp2;
				}
			}
			ret = true;
		}
		else if (d2<enterPichThr && d3<enterPichThr){
			actor->SetInteractMode(INTERACT_MODE::CHANGING_FORCE);
			preForce = f;
			prevPos = curPos;

			valRight = 2;
		}
		else{
			valRight = 0;
			float3 posMin, posMax;
			actor->GetPosRange(posMin, posMax);
			if (outOfDomain(curPos, posMin, posMax)){
				ret = false;
			}
			else{
				if (l->PointOnLensCenter3D(curPos, modelview, projection, winSize.x, winSize.y)){
					highlightingCenter = true;
					highlightingMajorSide = false;
					highlightingMinorSide = false;
					highlightingCuboidFrame = false;
				}
				else if (l->PointOnOuterBoundaryWallMajorSide3D(curPos, modelview, projection, winSize.x, winSize.y)){
					highlightingCenter = false;
					highlightingMajorSide = true;
					highlightingMinorSide = false;
					highlightingCuboidFrame = false;
				}
				else if (l->PointOnOuterBoundaryWallMinorSide3D(curPos, modelview, projection, winSize.x, winSize.y)){
					highlightingCenter = false;
					highlightingMajorSide = false;
					highlightingMinorSide = true;
					highlightingCuboidFrame = false;
				}
				else if (l->PointInCuboidRegion3D(curPos, modelview, projection, winSize.x, winSize.y)){
					highlightingCenter = false;
					highlightingMajorSide = false;
					highlightingMinorSide = false;
					highlightingCuboidFrame = true;
				}
				else{
					highlightingCenter = false;
					highlightingMajorSide = false;
					highlightingMinorSide = false;
					highlightingCuboidFrame = false;
				}
			}
			ret = true;
		}
		break;
		}
		case INTERACT_MODE::CHANGING_FORCE:
		{
			if (d2 > leavePinchThr || d3 > leavePinchThr){
				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				valRight = 0;
			}
			else{
				float3 c = l->c;
				f = preForce + (abs(dot(curPos - c, l->minorAxisGlobal)) - abs(dot(prevPos - c, l->minorAxisGlobal))) * 2 * 3;
				if (f < 0)
					f = 0;
				//send back new force
			}
			ret = true;
			break;
		}
		case INTERACT_MODE::MOVE_LENS:
		{
			if (d > leavePinchThr){
				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				(lenses->back())->justChanged = true;
			}
			else{
				float3 moveDir = curPos - prevPos;
				ChangeLensCenterbyTransferredLeap(lenses->back(), prevPointOfLens + moveDir);
			}
			ret = true;
			break;
		}
		case INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE:
		{
			if (d > leavePinchThr){
				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				(lenses->back())->justChanged = true;
			}
			else{
				float3 moveDir = curPos - prevPos;
				if (length(curPos - l->ctrlPoint3D1) < length(curPos - l->ctrlPoint3D2)){
					l->ctrlPoint3D1 = prevPointOfLens + moveDir;
				}
				else{
					l->ctrlPoint3D2 = prevPointOfLens + moveDir;
				}
			}
			ret = true;
			break;
		}
		case INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE:
		{
			float3 moveDir = curPos - prevPos;
			float3 pp = prevPointOfLens + moveDir;
			float newMinorDis = abs(dot(pp - l->c, l->minorAxisGlobal));

			if (d > leavePinchThr && newMinorDis > l->lSemiMinorAxisGlobal){
				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				(lenses->back())->justChanged = true;
				//highlightingMinorSide = false;
			}
			else{
				l->focusRatio = l->lSemiMinorAxisGlobal / newMinorDis;
			}
			ret = true;
			break;
		}
		default:
		{
			//do nothing
			ret = false;
		}
		}
	}



	
	leapFingerIndicators->numParticles = 1;
	leapFingerIndicators->pos[0] = markerPos;
	//if (ret){
	//	actor->blendOthers = true;
	//}
	//else{
	//	actor->blendOthers = false;
	//}
	return ret;

	/*
	//once we temporarily draw cursor together with lensRenderable, because arrowRenderable does not work with VR goggle
	lensRenderable->activedCursors = 1;
	lensRenderable->cursorPos[0] = make_float3(markerPos);
	lensRenderable->cursorColor[0] = leapFingerIndicators->val[0];*/
}

void LensLeapInteractor::SlotTwoHandChanged(float3 l, float3 r)
{
	if (lenses->size() > 0){
		if (LENS_TYPE::TYPE_CIRCLE == lenses->back()->type){
			ChangeLensCenterbyLeap(lenses->back(), (l + r) * 0.5);
			((CircleLens*)lenses->back())->radius = length(l - r) * 0.5;
			((CircleLens*)lenses->back())->justChanged = true;
		}
		actor->UpdateGL();
	}
}


void LensLeapInteractor::ChangeLensCenterbyTransferredLeap(Lens *l, float3 p)
{
	if (DEFORM_MODEL::OBJECT_SPACE == ((DeformGLWidget*)actor)->GetDeformModel()){

		float3 newCenter = p;

		//std::cout << "indexLeap " << p.x << " " << p.y << " " << p.z << std::endl;
		//std::cout << "newCenter " << newCenter.x << " " << newCenter.y << " " << newCenter.z << std::endl;

		if (l->type == LENS_TYPE::TYPE_LINE){
			float3 moveDir = newCenter - l->c;
			l->SetCenter(newCenter);
			((LineLens3D*)l)->ctrlPoint3D1 += moveDir;
			((LineLens3D*)l)->ctrlPoint3D2 += moveDir;
		}
		else if (l->type == LENS_TYPE::TYPE_CIRCLE){

		}
	}
	else{

	}

}


void LensLeapInteractor::ChangeLensCenterbyLeap(Lens *l, float3 p)
{
	if (DEFORM_MODEL::OBJECT_SPACE == ((DeformGLWidget*)actor)->GetDeformModel()){

		float3 newCenter = GetTransferredLeapPos(p);

		if (l->type == LENS_TYPE::TYPE_LINE){
			float3 moveDir = newCenter - l->c;
			l->SetCenter(newCenter);
			((LineLens3D*)l)->ctrlPoint3D1 += moveDir;
			((LineLens3D*)l)->ctrlPoint3D2 += moveDir;
		}
		else if (l->type == LENS_TYPE::TYPE_CIRCLE){

		}
	}
	else{
		int2 winSize = actor->GetWindowSize();
		GLfloat modelview[16];
		GLfloat projection[16];
		actor->GetModelview(modelview);
		actor->GetProjection(projection);
		float3 pScreen;
		float3 leapPos = GetNormalizedLeapPos(p);
		const float aa = 0.02f;
		float2 depthRange;
		((DeformGLWidget*)actor)->GetDepthRange(depthRange);

		bool usingVR = false;
		if (usingVR){
			pScreen.x = (1.0 - leapPos.x) * winSize.x;
			pScreen.y = clamp((1.0 - leapPos.z) * 2, 0.0f, 1.0f) * winSize.y;
			pScreen.z = depthRange.x + (depthRange.y - depthRange.x) * leapPos.y;
		}
		else{
			pScreen.x = leapPos.x * winSize.x;
			pScreen.y = leapPos.y * winSize.y;
			pScreen.z = depthRange.x + (depthRange.y - depthRange.x) * (1.0 - leapPos.z);
		}
		//std::cout << "depth:" << pScreen.z << std::endl;
		l->SetClipDepth(pScreen.z, modelview, projection);
		l->MoveLens(pScreen.x, pScreen.y, modelview, projection, winSize.x, winSize.y);
	}

}

float3 LensLeapInteractor::GetTransferredLeapPos(float3 p)
{
	////Xin's method when not using VR
	float3 leapPosNormalized = GetNormalizedLeapPos(p);

	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);

	float2 depthRange;
	((DeformGLWidget*)actor)->GetDepthRange(depthRange);

	float leapClipx = leapPosNormalized.x * 2 - 1;
	float leapClipy = leapPosNormalized.y * 2 - 1;
	float leapClipz = depthRange.x + (depthRange.y - depthRange.x) * (1.0 - leapPosNormalized.z);


	float _invmv[16];
	float _invpj[16];
	invertMatrix(projection, _invpj);
	invertMatrix(modelview, _invmv);

	return make_float3(Clip2ObjectGlobal(make_float4(leapClipx, leapClipy, leapClipz, 1.0), _invmv, _invpj));

}

