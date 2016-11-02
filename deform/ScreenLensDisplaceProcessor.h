#ifndef DISPLACE_H
#define DISPLACE_H
#include <thrust/device_vector.h>
#include "Processor.h"

class Lens;
class Particle;

class ScreenLensDisplaceProcessor:public Processor
{
	std::shared_ptr<Particle> particle;

	thrust::device_vector<float4> d_vec_posOrig;
	thrust::device_vector<float4> d_vec_posTarget;
	thrust::device_vector<float> d_vec_glyphSizeTarget;
	thrust::device_vector<float> d_vec_glyphBrightTarget;
	thrust::device_vector<int> d_vec_id;

	bool isRecomputeTargetNeeded = true;

	thrust::device_vector<char> feature;

	thrust::device_vector<float> d_vec_disToAim; //used for snapping
	std::vector<Lens*> *lenses;
	void InitFromParticle(std::shared_ptr<Particle> inputParticle);

public:
	ScreenLensDisplaceProcessor(std::vector<Lens*> *_lenses, std::shared_ptr<Particle> inputParticle)
	{
		lenses = _lenses;
		InitFromParticle(inputParticle);
	};
	
	void reset();

	void Compute(float* modelview, float* projection, int winW, int winH);


	bool process(float* modelview, float* projection, int winW, int winH) override;

	void LoadFeature(char* f, int num);
	void setRecomputeNeeded(){ isRecomputeTargetNeeded = true; }

	void DisplacePoints(std::vector<float2>& pts, std::vector<Lens*> lenses, float* modelview, float* projection, int winW, int winH); //used to draw the images of the deformed grid, used in Xin's PacificVis streamline paper

	float3 findClosetGlyph(float3 aim, int &snappedGlyphId);

};

#endif
