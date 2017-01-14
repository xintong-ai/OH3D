#include "GlyphInteractor.h"
#include "DeformGLWidget.h"
#include "glwidget.h"
#include "Particle.h"

void GlyphInteractor::mousePress(int x, int y, int modifier)
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