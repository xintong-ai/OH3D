#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

#include "GlyphRenderable.h"
#include "glwidget.h"
#include "DeformGLWidget.h"
#include "Particle.h""
#include "MeshDeformProcessor.h"
#include "screenLensDisplaceProcessor.h"

#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
//using namespace std;

#include <QOpenGLFunctions>
#include "ShaderProgram.h"

GlyphRenderable::GlyphRenderable(std::shared_ptr<Particle> _particle)
{
	particle = _particle;

	glyphSizeScale.assign(particle->numParticles, 1.0f);
	glyphBright.assign(particle->numParticles, 1.0f);
}

GlyphRenderable::~GlyphRenderable()
{ 
	if (nullptr != glProg){
		delete glProg;
		glProg = nullptr;
	}
}

void GlyphRenderable::ComputeDisplace(float _mv[16], float _pj[16])
{
	int winWidth, winHeight;
	actor->GetWindowSize(winWidth, winHeight);
	switch (((DeformGLWidget*)actor)->GetDeformModel())
	{
	case DEFORM_MODEL::SCREEN_SPACE:
	{
		if (screenLensDisplaceProcessor != 0){
			screenLensDisplaceProcessor->ProcessDeformation(&matrix_mv.v[0].x, &matrix_pj.v[0].x, winWidth, winHeight, &(particle->pos[0]), &glyphSizeScale[0], &glyphBright[0], isFreezingFeature, snappedGlyphId, snappedFeatureId);
		}
		break;
	}
	case DEFORM_MODEL::OBJECT_SPACE:
	{
		if (modelGrid != 0){
			modelGrid->ProcessParticleDeformation(&matrix_mv.v[0].x, &matrix_pj.v[0].x, winWidth, winHeight, particle, &glyphSizeScale[0], &glyphBright[0], isFreezingFeature, snappedGlyphId, snappedFeatureId);
		}
		break;
	}
	}
}

void GlyphRenderable::resetBrightness(){
	glyphBright.assign(particle->numParticles, 1.0);
};

void GlyphRenderable::mouseMove(int x, int y, int modifier)
{
}

void GlyphRenderable::resize(int width, int height)
{
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

	if (INTERACT_MODE::TRANSFORMATION == actor->GetInteractMode()) {
		if (screenLensDisplaceProcessor != 0){
			screenLensDisplaceProcessor->setRecomputeNeeded();
		}
		if (modelGrid != 0){
			modelGrid->setReinitiationNeed();
		}
	}
}


// !!! NOTE: result is not meaningful when no feature is loaded. Need to deal with this situation when calling this function. when no feature is loaded, return false 
bool GlyphRenderable::findClosetFeature(float3 aim, float3 & result, int & resid)
{
	/*
	///DO NOT DELETE!! WILL PROCESS LATER
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
	*/
	return false;
}



void GlyphRenderable::mousePress(int x, int y, int modifier)
{
	/*
	///DO NOT DELETE!!! will modify in the future
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
			std::cout << "pickedGlyph feature " << (int)(particle->feature[pickedGlyphId]) << std::endl;
			snappedGlyphId = pickedGlyphId;

			for (int i = 0; i < lenses->size(); i++) {
				Lens* l = (*lenses)[i];
				l->SetCenter(make_float3(particle->posOrig[snappedGlyphId]));
			}
		}

		isPickingGlyph = false;
		//emit glyphPickingFinished();
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
			for (int i = 0; i < lenses->size(); i++) {
				Lens* l = (*lenses)[i];
				l->SetCenter(featureCenter[snappedFeatureId - 1]);
			}
		}

		isPickingFeature = false;
		emit featurePickingFinished();
	}
	*/
}