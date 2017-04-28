//no need to redefine a VRRenderable if just drawing with basic opengl drawing functions.
//if using shader, need to redefine another VRRenderable


#ifndef VR_GLYPH_RENDERABLE_H
#define VR_GLYPH_RENDERABLE_H


#include "Renderable.h"
#include <memory>
class GlyphRenderable;
class ShaderProgram;
class VRWidget;
class VRGlyphRenderable : public Renderable
{
	Q_OBJECT
	
	GlyphRenderable* glyphRenderable;
	ShaderProgram* glProg;

protected:
	void init() override;
	void drawVR(float modelview[16], float projection[16], int eye) override;
	//void resize(int width, int height) override;

	VRWidget* vractor;
public:
	VRGlyphRenderable(GlyphRenderable* _glyphRenderable) { glyphRenderable = _glyphRenderable; }

	void SetVRActor(VRWidget* _a) override;
};
#endif