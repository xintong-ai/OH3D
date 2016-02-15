#ifndef VR_GLYPH_RENDERABLE_H
#define VR_GLYPH_RENDERABLE_H


#include "Renderable.h"
#include <memory>
class GlyphRenderable;
class VRGlyphRenderable : public Renderable
{
	Q_OBJECT
	
	GlyphRenderable* glyphRenderable;
protected:
	std::vector<float> glyphBright;
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	//void resize(int width, int height) override;

public:
	VRGlyphRenderable(GlyphRenderable* _glyphRenderable) { glyphRenderable = _glyphRenderable; }
};
#endif