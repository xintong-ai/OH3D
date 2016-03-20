#include "GlyphRenderable.h"
#include <Displace.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <glwidget.h>
#include <Lens.h>
#include <LensRenderable.h>



#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
using namespace std;

#include <QOpenGLFunctions>
#include "ShaderProgram.h"

void GlyphRenderable::ComputeDisplace()
{
	int2 winSize = actor->GetWindowSize();
	displace->Compute(&matrix_mv.v[0].x, &matrix_pj.v[0].x, winSize.x, winSize.y,
		((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses(), &pos[0], &glyphSizeScale[0], &glyphBright[0], isHighlightingFeature, snappedGlyphId, snappedFeatureId);
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
}

void GlyphRenderable::SetFeature(std::vector<char> & _feature, std::vector<float3> & _featureCenter)
{ 
	for (int i = 0; i < _feature.size(); i++) 
		feature[i] = _feature[i];
	displace->LoadFeature(&feature[0], feature.size());
	featureCenter = _featureCenter;
	//for (int i = 0; i < _featureCenter.size(); i++)
	//	featureCenter[i] = _featureCenter[i];
};

GlyphRenderable::~GlyphRenderable()
{ 
	if (nullptr != glProg){
		delete glProg;
		glProg = nullptr;
	}
}


void GlyphRenderable::RecomputeTarget()
{
	//displace->RecomputeTarget();
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
bool GlyphRenderable::findClosetFeature(float3 aim, float3 & result)
{
	int n = featureCenter.size();
	if (n < 1){
		return false;
	}

	int resid = -1;
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
	return true;
}



void GlyphRenderable::mousePress(int x, int y, int modifier)
{
	if (QApplication::keyboardModifiers() == Qt::AltModifier && isPicking){
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

		snappedGlyphId = cursorPixel[0] + cursorPixel[1] * 256 + cursorPixel[2] * 256 * 256 - 1;

		if (snappedGlyphId != -1){
			std::vector<Lens*> lenses = ((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses();
			for (int i = 0; i < lenses.size(); i++) {
				Lens* l = lenses[i];
				l->SetCenter(make_float3(posOrig[snappedGlyphId]));
			}
		}
	}
	else if (QApplication::keyboardModifiers() == Qt::ShiftModifier && isPickingFeature){
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

		snappedFeatureId = cursorPixel[0] + cursorPixel[1] * 256 + cursorPixel[2] * 256 * 256;

		//cout << snappedFeatureId << endl;

		if (snappedFeatureId > 0){
			std::vector<Lens*> lenses = ((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses();
			for (int i = 0; i < lenses.size(); i++) {
				Lens* l = lenses[i];
				l->SetCenter(featureCenter[snappedFeatureId-1]);
				//cout << featureCenter[snappedFeatureId - 1].x << " " << featureCenter[snappedFeatureId - 1].y << " " << featureCenter[snappedFeatureId - 1].z << endl;
			}
		}
	}
}