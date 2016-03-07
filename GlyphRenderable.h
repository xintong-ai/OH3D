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
	bool frameBufferObjectInitialized = false;

public:
	std::vector<float4> pos;
	std::vector<float4> posOrig;

	//used for feature rendering
	std::vector<char> feature;
	bool isHighlightingFeature = false;
	void SetFeature(std::vector<char> & _feature, std::vector<float3> & _featureCenter);
	std::vector<float3> featureCenter;

	bool isPickingFeature = true;
	int snappedFeatureId = -1;

	//used for picking and snapping
	bool isPicking = false;
	float3 findClosetGlyph(float3 aim);
	int GetSnappedGlyphId(){ return snappedGlyphId; }
	void SetSnappedGlyphId(int s){ snappedGlyphId = s; }


protected:
	std::shared_ptr<Displace> displace;
	std::vector<float> glyphSizeScale;
	std::vector<float> glyphBright;
	float glyphSizeAdjust = 0.5;
	ShaderProgram* glProg = nullptr;
	//bool displaceOn = true;
	void ComputeDisplace();
	void mouseMove(int x, int y, int modifier) override;
	void resize(int width, int height) override;
	GlyphRenderable(std::vector<float4>& _pos);

	//used for picking and snapping
	unsigned int vbo_vert_picking, vbo_indices_picking;
	int snappedGlyphId = -1;
	ShaderProgram *glPickingProg;
	unsigned int framebuffer, renderbuffer[2];
	void mousePress(int x, int y, int modifier) override;
	virtual void initPickingDrawingObjects() = 0;
	virtual void drawPicking(float modelview[16], float projection[16]) = 0;

public:
	~GlyphRenderable();
	void RecomputeTarget();
	void DisplacePoints(std::vector<float2>& pts);
	virtual void LoadShaders(ShaderProgram*& shaderProg) = 0;
	virtual void DrawWithoutProgram(float modelview[16], float projection[16], ShaderProgram* sp) = 0;
	//void SetDispalceOn(bool b) { displaceOn = b; }

public slots:
	void SlotGlyphSizeAdjustChanged(int v);
};
#endif