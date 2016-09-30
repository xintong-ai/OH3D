#ifndef DeformGlyph_RENDERABLE_H
#define DeformGlyph_RENDERABLE_H

#include "GlyphRenderable.h"
class DeformInterface;
class LineSplitModelGrid;
class Lens;
class ModelGrid;

class DeformGlyphRenderable: public GlyphRenderable
{
	/****timing****/
	StopWatchInterface *deformTimer = 0;
	int fpsCount = 0;        // FPS count for averaging
	int fpsLimit = 128;        // FPS limit for sampling
	void StartDeformTimer();
	void StopDeformTimer();
	bool displaceEnabled = true;
	
public:
	std::vector<Lens*> *lenses = 0; // a reference of the lenses, which is stored in LensRenderable now

	DeformGlyphRenderable(std::shared_ptr<Particle> _particle);

	~DeformGlyphRenderable();
	void RecomputeTarget();
	void ComputeDisplace(float _mv[16], float pj[16]);
	void SetModelGrid(LineSplitModelGrid* _modelGrid){ modelGrid = _modelGrid; }
	void SetModelGrid(ModelGrid* _modelGrid){ ; }

	void DisplacePoints(std::vector<float2>& pts);

	void SetDisplace(bool v){ displaceEnabled = v; }
	std::shared_ptr<DeformInterface> deformInterface;
	LineSplitModelGrid* modelGrid;
	float3 findClosetGlyph(float3 aim);
	void mousePress(int x, int y, int modifier) override;

protected:
	void init() override;
	void resize(int width, int height) override;


};

#endif