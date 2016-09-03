#ifndef GLYPH_RENDERABLE_H
#define GLYPH_RENDERABLE_H

#include "Renderable.h"
#include <memory>
class ShaderProgram;
class QOpenGLContext;
class StopWatchInterface;
//enum struct COLOR_MAP;
#include <ColorGradient.h>
#include <Particle.h>

class GlyphRenderable: public Renderable
{
	Q_OBJECT
	bool frameBufferObjectInitialized = false;

public:
	~GlyphRenderable();
	virtual void LoadShaders(ShaderProgram*& shaderProg) = 0;
	virtual void DrawWithoutProgram(float modelview[16], float projection[16], ShaderProgram* sp) = 0;
	//void SetDispalceOn(bool b) { displaceOn = b; }
	void SetGlyphSizeAdjust(float v){ glyphSizeAdjust = v; }

	int GetNumOfGlyphs(){ return particle->numParticles; }


	//used for feature freezing rendering
	bool isFreezingFeature = false;
	std::vector<char> feature;
	std::vector<float3> featureCenter;

	//used for feature snapping
	bool isPickingFeature = false;
	int GetSnappedFeatureId(){ return snappedFeatureId; }
	void SetSnappedFeatureId(int s){ snappedFeatureId = s; }
	bool findClosetFeature(float3 aim, float3 & result, int & resid);

	//used for picking and snapping
	bool isPickingGlyph = false;
	int GetSnappedGlyphId(){ return snappedGlyphId; }
	void SetSnappedGlyphId(int s){ snappedGlyphId = s; }

	virtual void resetColorMap(COLOR_MAP cm) = 0;

protected:
	std::shared_ptr<Particle> particle;

	float4* pos;
	//std::vector<float4> pos;
	
	std::vector<float> glyphSizeScale;
	std::vector<float> glyphBright;
	float glyphSizeAdjust = 1.0f;
	ShaderProgram* glProg = nullptr;
	//bool displaceOn = true;
	void mouseMove(int x, int y, int modifier) override;
	void resize(int width, int height) override;

	GlyphRenderable(std::shared_ptr<Particle> _particle);

	//used for picking and snapping
	unsigned int vbo_vert_picking, vbo_indices_picking;
	ShaderProgram *glPickingProg;
	unsigned int framebuffer, renderbuffer[2];
	virtual void initPickingDrawingObjects() = 0;
	virtual void drawPicking(float modelview[16], float projection[16], bool isForGlyph) = 0; //if isForGlyph=false, then it is for feature
	int snappedGlyphId = -1;
	int snappedFeatureId = -1;

public slots:
	void SlotGlyphSizeAdjustChanged(int v);

signals:
	void glyphPickingFinished();
	void featurePickingFinished();
};
#endif