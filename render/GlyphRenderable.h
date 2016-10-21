#ifndef GLYPH_RENDERABLE_H
#define GLYPH_RENDERABLE_H

#include "Renderable.h"
#include <memory>
class ShaderProgram;
class QOpenGLContext;
class StopWatchInterface;
class MeshDeformProcessor;
class ScreenLensDisplaceProcessor;
class PhysicalParticleDeformProcessor;

class Particle;
enum COLOR_MAP;

class GlyphRenderable: public Renderable
{
	Q_OBJECT

public:
	~GlyphRenderable();
	std::shared_ptr<PhysicalParticleDeformProcessor> physicalParticleDeformProcessor = 0;

	//used for deformation
	void SetScreenLensDisplaceComputer(std::shared_ptr<ScreenLensDisplaceProcessor> _screenLensDisplaceProcessor){ screenLensDisplaceProcessor = _screenLensDisplaceProcessor; }
	void SetModelGrid(std::shared_ptr<MeshDeformProcessor> _modelGrid){ meshDeformer = _modelGrid; }

	void mouseMove(int x, int y, int modifier) override;
	void resize(int width, int height) override;
	void mousePress(int x, int y, int modifier) override;

	virtual void setColorMap(COLOR_MAP cm, bool isReversed = false) {};
	bool colorByFeature = false;//when the particle has multi attributes or features, choose which attribute or color is used for color. currently a simple solution using bool

protected:
	GlyphRenderable(std::shared_ptr<Particle> _particle);

	std::shared_ptr<Particle> particle;

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


	//used for deformation
	void ComputeDisplace(float _mv[16], float pj[16]);
	std::shared_ptr<ScreenLensDisplaceProcessor> screenLensDisplaceProcessor = 0;
	std::shared_ptr<MeshDeformProcessor> meshDeformer = 0;

private:
	bool frameBufferObjectInitialized = false;

signals:
	void glyphPickingFinished();
	void featurePickingFinished();
};
#endif