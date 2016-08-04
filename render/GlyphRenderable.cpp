#include "GlyphRenderable.h"
#include <Displace.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <glwidget.h>
#include <Lens.h>
#include <LensRenderable.h>
#include <ModelGridRenderable.h>
#include <ModelGrid.h>
#include <TransformFunc.h>

#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
using namespace std;

#include <QOpenGLFunctions>
#include "ShaderProgram.h"
#include <helper_timer.h>

void GlyphRenderable::StartDeformTimer()
{
#if ENABLE_TIMER
	sdkStartTimer(&deformTimer);
#endif
}

void GlyphRenderable::StopDeformTimer()
{
#if ENABLE_TIMER
	sdkStopTimer(&deformTimer);
	fpsCount++;
	if (fpsCount == fpsLimit)
	{
		qDebug() << "Deform time (ms):\t" << sdkGetAverageTimerValue(&deformTimer);
		fpsCount = 0;
		sdkResetTimer(&deformTimer);
	}
#endif
}

void GlyphRenderable::ComputeDisplace(float _mv[16], float _pj[16])
{
	if (!displaceEnabled) return;
	StartDeformTimer();
	int2 winSize = actor->GetWindowSize();
	switch (actor->GetDeformModel())
	{
	case DEFORM_MODEL::SCREEN_SPACE:
	{
		displace->Compute(&matrix_mv.v[0].x, &matrix_pj.v[0].x, winSize.x, winSize.y,
			((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses(), &pos[0], &glyphSizeScale[0], &glyphBright[0], isFreezingFeature, snappedGlyphId, snappedFeatureId);
		break;
	}
	case DEFORM_MODEL::OBJECT_SPACE:
	{
		if (((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses().size() < 1)
			return;

		Lens *l = ((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses().back();
		
		float focusRatio = ((LensRenderable*)actor->GetRenderable("lenses"))->GetBackLensFocusRatio();

		//convert the camera location from camera space to object space
		//https://www.opengl.org/archives/resources/faq/technical/viewing.htm
		QMatrix4x4 q_modelview = QMatrix4x4(_mv);
		q_modelview = q_modelview.transposed();
		QMatrix4x4 q_inv_modelview = q_modelview.inverted();

		QVector4D cameraObj = q_inv_modelview * QVector4D(0, 0, 0, 1);
		cameraObj = cameraObj / cameraObj.w();
		float3 lensCen = ((LensRenderable*)actor->GetRenderable("lenses"))->GetBackLensCenter();

		float3 lensDir = make_float3(
			cameraObj.x() - lensCen.x,
			cameraObj.y() - lensCen.y,
			cameraObj.z() - lensCen.z);
		lensDir = normalize(lensDir);

		if (l->type == TYPE_CIRCLE && modelGrid->gridType == UNIFORM_GRID){
			float radius = ((LensRenderable*)actor->GetRenderable("lenses"))->GetBackLensObjectRadius();

			modelGrid->Update(&lensCen.x, &lensDir.x, focusRatio, radius);
			modelGrid->UpdatePointCoords(&pos[0], pos.size());

			const float dark = 0.1;
			const float transRad = radius / focusRatio;
			for (int i = 0; i < pos.size(); i++) {
				float3 vert = make_float3(pos[i]);
				//float3 lensCenFront = lensCen + lensDir * radius;
				float3 lensCenBack = lensCen - lensDir * radius;
				float3 lensCenFront2Vert = vert - lensCenBack;
				float lensCenFront2VertProj = dot(lensCenFront2Vert, lensDir);
				float3 moveVec = lensCenFront2Vert - lensCenFront2VertProj * lensDir;
				glyphBright[i] = 1.0;
				if (lensCenFront2VertProj < 0){
					float dist2Ray = length(moveVec);
					if (dist2Ray < radius / focusRatio){
						glyphBright[i] = std::max(dark, 1.0f / (0.5f * (-lensCenFront2VertProj) + 1.0f));;
					}
				}
			}

		}
		else if (l->type == TYPE_LINE && modelGrid->gridType == LINESPLIT_UNIFORM_GRID){

			//LineLens has info on the screen space. need to compute info on the world space
			//the following computation code is temporarily placed here. a better choice is to put them in lens.cpp, but more design of Lens.h is needed

			//screen length to object
			int winWidth, winHeight;
			actor->GetWindowSize(winWidth, winHeight);
			
			//transfer the end points of the major and minor axis to global space
			float2 centerScreen = l->GetCenterScreenPos(_mv, _pj, winWidth, winHeight);
			float2 endPointSemiMajorAxisScreen = centerScreen + ((LineLens*)l)->lSemiMajorAxis*((LineLens*)l)->direction;
			float2 endPointSemiMinorAxisScreen = centerScreen + ((LineLens*)l)->lSemiMinorAxis*make_float2(-((LineLens*)l)->direction.y, ((LineLens*)l)->direction.x);
				
			float4 centerInClip = Object2Clip(make_float4(l->c, 1), _mv, _pj);

			//_prj means the point's projection on the plane that has the same z clip-space coord with the lens center
			float3 endPointSemiMajorAxisClip_prj = make_float3(Screen2Clip(endPointSemiMajorAxisScreen, winWidth, winHeight), centerInClip.z);
			float3 endPointSemiMinorAxisClip_prj = make_float3(Screen2Clip(endPointSemiMinorAxisScreen, winWidth, winHeight), centerInClip.z);

			QMatrix4x4 q_projection = QMatrix4x4(_pj);
			q_projection = q_projection.transposed();
			QMatrix4x4 q_inv_projection = q_projection.inverted();

			float *_invmv = q_inv_modelview.data();
			float *_inpj = q_inv_projection.data();
			float3 endPointSemiMajorAxisGlobal_prj = make_float3(Clip2ObjectGlobal(make_float4(endPointSemiMajorAxisClip_prj, 1), _invmv, _inpj));
			float3 endPointSemiMinorAxisGlobal_prj = make_float3(Clip2ObjectGlobal(make_float4(endPointSemiMinorAxisClip_prj, 1), _invmv, _inpj));

	
			//using the end points of the major and minor axis in global space, to compute the length and direction of major and minor axis in global space
			float lSemiMajorAxisGlobal_prj = length(endPointSemiMajorAxisGlobal_prj - l->c);
			float lSemiMinorAxisGlobal_prj = length(endPointSemiMinorAxisGlobal_prj - l->c);
			float3 majorAxisGlobal_prj = normalize(endPointSemiMajorAxisGlobal_prj - l->c);
			float3 minorAxisGlobal_prj = normalize(endPointSemiMinorAxisGlobal_prj - l->c);

			float3 minorAxisGlobal = normalize(cross(lensDir, majorAxisGlobal_prj));
			float3 majorAxisGlobal = normalize(cross(minorAxisGlobal, lensDir));
			float lSemiMajorAxisGlobal = lSemiMajorAxisGlobal_prj / dot(majorAxisGlobal, majorAxisGlobal_prj);
			float lSemiMinorAxisGlobal = lSemiMinorAxisGlobal_prj / dot(minorAxisGlobal, minorAxisGlobal_prj);


			modelGrid->ReinitiateMesh(l->c, lSemiMajorAxisGlobal, lSemiMinorAxisGlobal, majorAxisGlobal, ((LineLens*)l)->focusRatio, lensDir, &posOrig[0], pos.size());

			modelGrid->Update(&lensCen.x, &lensDir.x, lSemiMajorAxisGlobal, lSemiMinorAxisGlobal, focusRatio, majorAxisGlobal);
			//modelGrid->UpdatePointCoords(&pos[0], pos.size(), &posOrig[0]);
			//modelGrid->UpdatePointCoords_LineMeshLens_Thrust(&pos[0], pos.size());
			modelGrid->UpdatePointCoordsAndBright_LineMeshLens_Thrust(&pos[0], &glyphBright[0], pos.size(), l->c, lSemiMajorAxisGlobal, lSemiMinorAxisGlobal, majorAxisGlobal, ((LineLens*)l)->focusRatio, lensDir, isFreezingFeature, snappedFeatureId, &feature[0]);
		}
		break;
	}
	}
	StopDeformTimer();
}

void GlyphRenderable::init()
{
	modelGrid->InitGridDensity(&pos[0], pos.size());
}

GlyphRenderable::GlyphRenderable(std::vector<float4>& _pos)
{ 
	pos = _pos; 
	posOrig = pos;
	feature.resize(pos.size());

	displace = std::make_shared<Displace>();
	displace->LoadOrig(&pos[0], pos.size());
	glyphSizeScale.assign(pos.size(), 1.0f);
	glyphBright.assign(pos.size(), 1.0f);

	sdkCreateTimer(&deformTimer);
}

void GlyphRenderable::SetFeature(std::vector<char> & _feature, std::vector<float3> & _featureCenter)
{ 
	for (int i = 0; i < _feature.size(); i++) 
		feature[i] = _feature[i];
	displace->LoadFeature(&feature[0], feature.size());
	featureCenter = _featureCenter;
};

GlyphRenderable::~GlyphRenderable()
{ 
	sdkDeleteTimer(&deformTimer);
	if (nullptr != glProg){
		delete glProg;
		glProg = nullptr;
	}
}


void GlyphRenderable::RecomputeTarget()
{
	if (!visible)
		return;

	switch (actor->GetDeformModel())
	{
	case DEFORM_MODEL::SCREEN_SPACE:
		displace->RecomputeTarget();
		break;
	}
}

void GlyphRenderable::DisplacePoints(std::vector<float2>& pts)
{
	int2 winSize = actor->GetWindowSize();
	//displace->DisplacePoints(pts,
	//	((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses(), &matrix_mv.v[0].x, &matrix_pj.v[0].x, winSize.x, winSize.y);
}

void GlyphRenderable::SlotGlyphSizeAdjustChanged(int v)
{
	glyphSizeAdjust = v * 0.1;
	actor->UpdateGL();
}

void GlyphRenderable::mouseMove(int x, int y, int modifier)
{
}

void GlyphRenderable::resize(int width, int height)
{
	if (INTERACT_MODE::TRANSFORMATION == actor->GetInteractMode()) {
		RecomputeTarget();
	}

	if (!frameBufferObjectInitialized){
		qgl->glGenRenderbuffers(2, renderbuffer);
		qgl->glGenFramebuffers(1, &framebuffer);
		frameBufferObjectInitialized = true;
	}

	qgl->glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer[0]);
	qgl->glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, width, height);

	qgl->glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer[1]);
	qgl->glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);


	qgl->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
	qgl->glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
		GL_RENDERBUFFER, renderbuffer[0]);
	qgl->glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
		GL_RENDERBUFFER, renderbuffer[1]);
	qgl->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

float3 GlyphRenderable::findClosetGlyph(float3 aim)
{
	return displace->findClosetGlyph(aim, snappedGlyphId);

}

// !!! NOTE: result is not meaningful when no feature is loaded. Need to deal with this situation when calling this function. when no feature is loaded, return false 
bool GlyphRenderable::findClosetFeature(float3 aim, float3 & result, int & resid)
{
	int n = featureCenter.size();
	if (n < 1){
		return false;
	}

	resid = -1;
	float resDistance = 9999999999;
	result = make_float3(0, 0, 0);
	for (int i=0; i < n; i++){
		float curRes = length(aim - featureCenter[i]);
		if (curRes < resDistance){
			resid = i;
			resDistance = curRes;
			result = featureCenter[i];
		}
	}

	snappedFeatureId = resid + 1;
	resid = snappedFeatureId;
	return true;
}



void GlyphRenderable::mousePress(int x, int y, int modifier)
{
	if (isPickingGlyph){
		qgl->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		float modelview[16], projection[16];
		actor->GetModelview(modelview);
		actor->GetProjection(projection);

		drawPicking(modelview, projection, true);

		qgl->glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
		unsigned char cursorPixel[4];
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, cursorPixel);

		int pickedGlyphId = cursorPixel[0] + cursorPixel[1] * 256 + cursorPixel[2] * 256 * 256 - 1;
		
		if (pickedGlyphId != -1){
			snappedGlyphId = pickedGlyphId;
			std::vector<Lens*> lenses = ((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses();
			for (int i = 0; i < lenses.size(); i++) {
				Lens* l = lenses[i];
				l->SetCenter(make_float3(posOrig[snappedGlyphId]));
			}
		}

		isPickingGlyph = false;
		emit glyphPickingFinished();
	}
	else if (isPickingFeature){
		qgl->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		float modelview[16], projection[16];
		actor->GetModelview(modelview);
		actor->GetProjection(projection);

		drawPicking(modelview, projection, false);

		qgl->glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
		unsigned char cursorPixel[4];
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, cursorPixel);

		int pickedGlyphId = cursorPixel[0] + cursorPixel[1] * 256 + cursorPixel[2] * 256 * 256;

		if (pickedGlyphId > -1){
			if (feature[pickedGlyphId] > 0){
				snappedFeatureId = feature[pickedGlyphId];
				std::vector<Lens*> lenses = ((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses();
				for (int i = 0; i < lenses.size(); i++) {
					Lens* l = lenses[i];
					if (featureCenter.size()>0)
						l->SetCenter(featureCenter[snappedFeatureId - 1]);
					else{
						l->SetCenter(make_float3(posOrig[pickedGlyphId]));
					}
				}
			}
		}

		isPickingFeature = false;
		emit featurePickingFinished();
	}
}