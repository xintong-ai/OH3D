#include <DeformInterface.h>
#include <DeformGlyphRenderable.h>
#include <LineSplitModelGrid.h>
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

void DeformGlyphRenderable::init()
{
	//modelGrid->InitGridDensity(&pos[0], particle->numParticles);
}


void DeformGlyphRenderable::ComputeDisplace(float _mv[16], float _pj[16])
{
	if (!displaceEnabled) return;


	if (lenses == 0 || lenses->size() <1 || modelGrid == 0)
		return;

	//should remove the dependence on LensRenderable later. use the reference "lenses" instead
	
	Lens *l = (*lenses)[lenses->size() - 1];

	if (l->justChanged){

		switch (((DeformGLWidget*)actor)->GetDeformModel())
		{
		case DEFORM_MODEL::SCREEN_SPACE:
		{
			deformInterface->RecomputeTarget();
			l->justChanged = false;
			break;
		}
		case DEFORM_MODEL::OBJECT_SPACE:
		{
			if (l->type == TYPE_LINE)
				modelGrid->setReinitiationNeed();
			l->justChanged = false;
			break;
		}
		}
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

			if (l->type == TYPE_LINE && modelGrid->gridType == LINESPLIT_UNIFORM_GRID && !l->isConstructing){
				QMatrix4x4 q_modelview = QMatrix4x4(_mv);
				q_modelview = q_modelview.transposed();
				QMatrix4x4 q_inv_modelview = q_modelview.inverted();

				QVector4D cameraObj = q_inv_modelview * QVector4D(0, 0, 0, 1);
				cameraObj = cameraObj / cameraObj.w();
			
				int winWidth, winHeight;
				actor->GetWindowSize(winWidth, winHeight);

				((LineLens3D*)l)->UpdateLineLensGlobalInfo(make_float3(cameraObj.x(), cameraObj.y(), cameraObj.z()), winWidth, winHeight, _mv, _pj, particle->posMin, particle->posMax);


				modelGrid->ReinitiateMeshForParticle((LineLens3D*)l, particle.get());

				if (l->justMoved) 
				{
					////the related work needs more time to finish. To keep the lens facing the camera, the lens nodes needs to be rotated. Also the lens region may need to change to guarantee to cover the whole region
					//modelGrid->MoveMesh(l->moveVec);
					l->justMoved = false;
				}

				modelGrid->UpdateMesh(&(((LineLens3D*)l)->c.x), &(((LineLens3D*)l)->lensDir.x), ((LineLens3D*)l)->lSemiMajorAxisGlobal, ((LineLens3D*)l)->lSemiMinorAxisGlobal, ((LineLens3D*)l)->focusRatio, ((LineLens3D*)l)->majorAxisGlobal);

				modelGrid->UpdatePointCoordsAndBright_LineMeshLens_Thrust(particle.get(), &glyphBright[0], (LineLens3D*)l, isFreezingFeature, snappedFeatureId);
				
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