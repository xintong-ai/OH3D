#ifndef DeformGlyph_RENDERABLE_H
#define DeformGlyph_RENDERABLE_H

#include "GlyphRenderable.h"

class LineSplitModelGrid;
class Lens;
class ModelGrid;
class Displace;
class DeformGlyphRenderable: public GlyphRenderable
{
	/****timing****/
	StopWatchInterface *deformTimer = 0;
	int fpsCount = 0;        // FPS count for averaging
	int fpsLimit = 128;        // FPS limit for sampling
	void StartDeformTimer();
	void StopDeformTimer();
	
public:
	std::vector<Lens*> *lenses = 0; // a reference of the lenses, which is stored in LensRenderable now

	DeformGlyphRenderable(std::shared_ptr<Particle> _particle);

	~DeformGlyphRenderable();
	void ComputeDisplace(float _mv[16], float pj[16]);
	void SetModelGrid(LineSplitModelGrid* _modelGrid){ modelGrid = _modelGrid; }
	void SetScreenLensDisplaceComputer(std::shared_ptr<Displace> _screenLensDisplaceProcessor){ screenLensDisplaceProcessor = _screenLensDisplaceProcessor; }

	void DisplacePoints(std::vector<float2>& pts);

	std::shared_ptr<Displace> screenLensDisplaceProcessor = 0;
	LineSplitModelGrid* modelGrid;
	float3 findClosetGlyph(float3 aim);
	void mousePress(int x, int y, int modifier) override;

protected:
	void init() override;
	void resize(int width, int height) override;


};

#endif