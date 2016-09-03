#include "GlyphRenderable.h"
//#include <Displace.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <glwidget.h>
#include <helper_math.h>

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
	pos = &(particle->pos[0]);

	feature = particle->feature;

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


