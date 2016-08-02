#include "GlyphRenderable.h"
class DeformInterface;
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
	DeformGlyphRenderable(std::vector<float4>& _pos);
	~DeformGlyphRenderable();
	void RecomputeTarget();
	void ComputeDisplace(float _mv[16], float pj[16]);
	void SetModelGrid(ModelGrid* _modelGrid){ modelGrid = _modelGrid; }

	void DisplacePoints(std::vector<float2>& pts);
	void SetFeature(std::vector<char> & _feature, std::vector<float3> & _featureCenter);


	void EnableDisplace(bool v){ displaceEnabled = v; }
	std::shared_ptr<DeformInterface> deformInterface;
	ModelGrid* modelGrid;
	float3 findClosetGlyph(float3 aim);
	void mousePress(int x, int y, int modifier) override;

protected:
	void init() override;
	void resize(int width, int height) override;


};