#ifndef PARTICLE_H
#define PARTICLE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include <math.h>
#include <memory>
//using namespace std;

#include <cuda_runtime.h>
#include <helper_cuda.h>

class Particle
{
public:
	std::vector<float4> posOrig;
	float3 posMin, posMax;
	int numParticles = 0;

	std::vector<float4> pos; //current position after deformation

	std::vector<float3> orientation; //orientation of each particle. used when the shape of each particle is not isotropic

	//to record the attribute(s) that might be important, two ways can be used
	//the first case is for only one attribute
	std::vector<float> val; 
	float valMin, valMax;
	//the second case is for multiple attribute
	std::vector<float> valTuple;
	int tupleCount;
	//when one way of attribute recording is used, the other one can be ignored

	bool hasFeature = false;
	std::vector<char> feature; //actually is segmentation tag. should be named as "label" or "tag"
	void setFeature(std::vector<char> _f);
	char featureMin, featureMax;

	Particle(){};
	Particle(std::vector<float4>  &_pos, std::vector<float>  &_val);
	
	~Particle()
	{
	};

	void init(std::vector<float4> &_pos, std::vector<float> & _val);

	void updateMaxMinValAndPos();
	void clear(){};
	
	void featureReshuffle();
	void reset();

	//variables for rendering. are not needed if not used for a glyphRenderable
	//generally rendering
	bool hasInitedForRendering = false;
	void initForRendering(float s = 1.0f, float b = 1.0f);
	std::vector<float> glyphBright;
	std::vector<float> glyphSizeScale;
	//used for feature freezing / snapping
	bool isFreezingFeature = false;
	bool isPickingFeature = false;
	int snappedGlyphId = -1;
	int snappedFeatureId = -1;
	int GetSnappedFeatureId(){ return snappedFeatureId; }
	void SetSnappedFeatureId(int s){ snappedFeatureId = s; }
	bool findClosetFeature(float3 aim, float3 & result, int & resid);
	//used for picking and snapping
	bool isPickingGlyph = false;
	int GetSnappedGlyphId(){ return snappedGlyphId; }
	void SetSnappedGlyphId(int s){ snappedGlyphId = s; }
	
	void extractOrientation(int id); //special function for ImmersiveDeformParticle and ImmersiveDeformParticleTV
private:

};

#endif