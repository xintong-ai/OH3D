#ifndef GLYPH_RENDERABLE_H
#define GLYPH_RENDERABLE_H

#include "Renderable.h"
#include <memory>
class ShaderProgram;
class QOpenGLContext;
class StopWatchInterface;
class LineSplitModelGrid;
class ScreenLensDisplaceProcessor;
class Particle;
enum COLOR_MAP;

class GlyphRenderable: public Renderable
{
	Q_OBJECT

public:
	~GlyphRenderable();

	//used for deformation
	void SetScreenLensDisplaceComputer(std::shared_ptr<ScreenLensDisplaceProcessor> _screenLensDisplaceProcessor){ screenLensDisplaceProcessor = _screenLensDisplaceProcessor; }
	void SetModelGrid(std::shared_ptr<LineSplitModelGrid> _modelGrid){ modelGrid = _modelGrid; }

	void mouseMove(int x, int y, int modifier) override;
	void resize(int width, int height) override;
	void mousePress(int x, int y, int modifier) override;

	virtual void setColorMap(COLOR_MAP cm, bool isReversed = false) = 0;
	bool colorByFeature = false;//when the particle has multi attributes or features, choose which attribute or color is used for color. currently a simple solution using bool
	void SetGlyphSizeAdjust(float v){ glyphSizeAdjust = v; }
	void resetBrightness();

	//used for feature freezing / snapping
	bool isFreezingFeature = false;
	bool isPickingFeature = false;
	int GetSnappedFeatureId(){ return snappedFeatureId; }
	void SetSnappedFeatureId(int s){ snappedFeatureId = s; }
	bool findClosetFeature(float3 aim, float3 & result, int & resid);

	//used for picking and snapping
	bool isPickingGlyph = false;
	int GetSnappedGlyphId(){ return snappedGlyphId; }
	void SetSnappedGlyphId(int s){ snappedGlyphId = s; }

protected:
	GlyphRenderable(std::shared_ptr<Particle> _particle);

	std::shared_ptr<Particle> particle;

	std::vector<float> glyphBright;
	//both are used for adjust size
	std::vector<float> glyphSizeScale;
	float glyphSizeAdjust = 1.0f;
	
	//used for drawing
	ShaderProgram* glProg = nullptr;
	virtual void LoadShaders(ShaderProgram*& shaderProg) = 0;
	virtual void DrawWithoutProgram(float modelview[16], float projection[16], ShaderProgram* sp) = 0;

	//used for picking and snapping
	unsigned int vbo_vert_picking, vbo_indices_picking;
	ShaderProgram *glPickingProg;
	unsigned int framebuffer, renderbuffer[2];
	virtual void initPickingDrawingObjects() = 0;
	virtual void drawPicking(float modelview[16], float projection[16], bool isForGlyph) = 0; //if isForGlyph=false, then it is for feature
	int snappedGlyphId = -1;
	int snappedFeatureId = -1;

	//used for deformation
	void ComputeDisplace(float _mv[16], float pj[16]);
	std::shared_ptr<ScreenLensDisplaceProcessor> screenLensDisplaceProcessor = 0;
	std::shared_ptr<LineSplitModelGrid> modelGrid = 0;

private:
	bool frameBufferObjectInitialized = false;

signals:
	void glyphPickingFinished();
	void featurePickingFinished();
};
#endif