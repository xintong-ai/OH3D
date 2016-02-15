#ifndef GLYPH_RENDERABLE_H
#define GLYPH_RENDERABLE_H

#include "Renderable.h"
#include <memory>
class Displace;
class ShaderProgram;
class QOpenGLContext;
class GlyphRenderable: public Renderable
{
	Q_OBJECT
protected:
	std::vector<float4> pos;
	std::shared_ptr<Displace> displace;
	std::vector<float> glyphSizeScale;
	std::vector<float> glyphBright;
	float glyphSizeAdjust = 0.5;
	ShaderProgram* glProg;
	//bool displaceOn = true;
	void ComputeDisplace();
	GlyphRenderable(std::vector<float4>& _pos);
	void mouseMove(int x, int y, int modifier) override;
	void resize(int width, int height) override;

public:
	void RecomputeTarget();
	void DisplacePoints(std::vector<float2>& pts);
	virtual void LoadShaders(ShaderProgram*& shaderProg){}
	virtual void DrawWithoutProgram(float modelview[16], float projection[16], QOpenGLContext* ctx, ShaderProgram* sp) {}
	//void SetDispalceOn(bool b) { displaceOn = b; }
public slots:
	void SlotGlyphSizeAdjustChanged(int v);
};
#endif