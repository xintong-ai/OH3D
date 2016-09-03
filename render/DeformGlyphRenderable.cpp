#include <DeformInterface.h>
#include <DeformGlyphRenderable.h>
#include <ModelGrid.h>
#include <ModelGridRenderable.h>
#include <helper_timer.h>
#include <DeformGLWidget.h>
#include <LensRenderable.h>
#include <Lens.h>
#include <TransformFunc.h>


#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()


DeformGlyphRenderable::DeformGlyphRenderable(std::shared_ptr<Particle> _particle)
:GlyphRenderable(_particle)
{
	deformInterface = std::make_shared<DeformInterface>();
	deformInterface->LoadOrig(&pos[0], particle->numParticles);
	sdkCreateTimer(&deformTimer);
}

DeformGlyphRenderable::~DeformGlyphRenderable()
{
	sdkDeleteTimer(&deformTimer);
}


void DeformGlyphRenderable::DisplacePoints(std::vector<float2>& pts)
{
	int2 winSize = actor->GetWindowSize();
	//deformInterface->DisplacePoints(pts,
	//	((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses(), &matrix_mv.v[0].x, &matrix_pj.v[0].x, winSize.x, winSize.y);
}


void DeformGlyphRenderable::RecomputeTarget()
{
	if (!visible)
		return;

	switch (((DeformGLWidget*)actor)->GetDeformModel())
	{
	case DEFORM_MODEL::SCREEN_SPACE:
		deformInterface->RecomputeTarget();
		break;
	}
}

void DeformGlyphRenderable::StartDeformTimer()
{
#if ENABLE_TIMER
	sdkStartTimer(&deformTimer);
#endif
}


void DeformGlyphRenderable::StopDeformTimer()
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


void DeformGlyphRenderable::SetFeature(std::vector<char> & _feature, std::vector<float3> & _featureCenter)
{
	for (int i = 0; i < _feature.size(); i++)
		feature[i] = _feature[i];
	deformInterface->LoadFeature(&feature[0], feature.size());
	featureCenter = _featureCenter;
};


void DeformGlyphRenderable::init()
{
	modelGrid->InitGridDensity(&pos[0], particle->numParticles);
}


void DeformGlyphRenderable::ComputeDisplace(float _mv[16], float _pj[16])
{
	if (!displaceEnabled) return;

	//should remove the dependence on LensRenderable later. use the reference "lenses" instead
	if (((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses().size() < 1)
		return;

	if (((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses().back()->justChanged){
		
		switch (((DeformGLWidget*)actor)->GetDeformModel())
		{
		case DEFORM_MODEL::SCREEN_SPACE:
			deformInterface->RecomputeTarget();
			break;
		}
		
		((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses().back()->justChanged = false;
		//this setting can only do deform based on the last lens
	}

	
	StartDeformTimer();
	int2 winSize = actor->GetWindowSize();
	switch (((DeformGLWidget*)actor)->GetDeformModel())
	{
	case DEFORM_MODEL::SCREEN_SPACE:
	{
		deformInterface->Compute(&matrix_mv.v[0].x, &matrix_pj.v[0].x, winSize.x, winSize.y,
			((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses(), &pos[0], &glyphSizeScale[0], &glyphBright[0], isFreezingFeature, snappedGlyphId, snappedFeatureId);
		break;
	}
	case DEFORM_MODEL::OBJECT_SPACE:
	{
		Lens *l = ((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses().back();
		
		if (l->type == TYPE_LINE && modelGrid->gridType == LINESPLIT_UNIFORM_GRID){
			QMatrix4x4 q_modelview = QMatrix4x4(_mv);
			q_modelview = q_modelview.transposed();
			QMatrix4x4 q_inv_modelview = q_modelview.inverted();

			QVector4D cameraObj = q_inv_modelview * QVector4D(0, 0, 0, 1);
			cameraObj = cameraObj / cameraObj.w();
			
			int winWidth, winHeight;
			actor->GetWindowSize(winWidth, winHeight);

			((LineLens*)l)->UpdateLineLensGlobalInfo(make_float3(cameraObj.x(), cameraObj.y(), cameraObj.z()), winWidth, winHeight, _mv, _pj);

			modelGrid->ReinitiateMeshForParticle((LineLens*)l, particle.get());

			modelGrid->Update(&(((LineLens*)l)->c.x), &(((LineLens*)l)->lensDir.x), ((LineLens*)l)->lSemiMajorAxisGlobal, ((LineLens*)l)->lSemiMinorAxisGlobal, ((LineLens*)l)->focusRatio, ((LineLens*)l)->majorAxisGlobal);

			modelGrid->UpdatePointCoordsAndBright_LineMeshLens_Thrust(&pos[0], &glyphBright[0], particle->numParticles, (LineLens*)l, isFreezingFeature, snappedFeatureId, &feature[0]);

		}
		else if (l->type == TYPE_CIRCLE && modelGrid->gridType == UNIFORM_GRID){
			//TODO
			;
		}

		break;
	}
	}
	StopDeformTimer();
}

void DeformGlyphRenderable::resize(int width, int height)
{
	if (INTERACT_MODE::TRANSFORMATION == actor->GetInteractMode()) {
		RecomputeTarget();
	}
	GlyphRenderable::resize(width, height);
}


float3 DeformGlyphRenderable::findClosetGlyph(float3 aim)
{
	return deformInterface->findClosetGlyph(aim, snappedGlyphId);

}

void DeformGlyphRenderable::mousePress(int x, int y, int modifier)
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
				l->SetCenter(make_float3(particle->posOrig[snappedGlyphId]));
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

		if (pickedGlyphId > 0){
			snappedFeatureId = pickedGlyphId;
			std::vector<Lens*> lenses = ((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses();
			for (int i = 0; i < lenses.size(); i++) {
				Lens* l = lenses[i];
				l->SetCenter(featureCenter[snappedFeatureId - 1]);
			}
		}

		isPickingFeature = false;
		emit featurePickingFinished();
	}
}