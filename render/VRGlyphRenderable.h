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
	std::vector<float> glyphBright;
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	//void resize(int width, int height) override;

	VRWidget* vractor;
public:
	VRGlyphRenderable(GlyphRenderable* _glyphRenderable) { glyphRenderable = _glyphRenderable; }

	void SetVRActor(VRWidget* _a) override;
};
#endif