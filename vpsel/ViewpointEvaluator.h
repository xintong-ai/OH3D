#ifndef VIEWPOINTEVALUATOR_H
#define VIEWPOINTEVALUATOR_H

#define _USE_MATH_DEFINES
#include <cmath>
#include "myDefine.h"
#include <memory>
#include <vector>

#include "Volume.h"

class Particle;

enum VPMethod{
	BS05,
	JS06Sphere,
	Tao09Detail,
	LabelVisibility
};

struct SpherePoint {
	//float info[2];//info[0]:lat, info[1]:lon
	float3 p;
	
	SpherePoint(){};

	SpherePoint(float _lat, float _lon)
	{
		p.x = cos(_lat)*cos(_lon);
		p.y = cos(_lat)*sin(_lon);
		p.z = sin(_lat);
	}
};

class Volume;
class VolumeCUDA;
class ViewpointEvaluator
{
public:
	ViewpointEvaluator(std::shared_ptr<RayCastingParameters> _r, std::shared_ptr<Volume> v);
	~ViewpointEvaluator(){
		if (d_r != 0){
			cudaFree(d_r); d_r = 0;
		};
		for (int i = 0; i < cubeFaceHists.size(); i++){
			if (cubeFaceHists[i] != 0){
				cudaFree(cubeFaceHists[i]);
			}
		}
	};

	VPMethod currentMethod = Tao09Detail;

	void setSpherePoints(int n = 512);
	void setLabel(std::shared_ptr<VolumeCUDA> labelVol);
	void initDownSampledResultVolume(int3 sampleSize);	
	void compute_UniformSampling(VPMethod m);
	void compute_SkelSampling(VPMethod m);
	void saveResultVol(const char*);

	float maxEntropy;
	float3 optimalEyeInLocal;

	std::vector<float> cubeInfo;
	void computeCubeEntropy(float3 eyeInLocal, float3 viewDir, float3 upDir, VPMethod m);

	std::vector<std::shared_ptr<Particle>> skelViews;
	
	bool useHist = true;  //most papers do not use histogram to compute entropy. however we mostly use histogram. if true, each bin will be computed a probability; if false, each pixel will be computed a probability
	int maxLabel = 1; //!! data dependant

private:
	std::shared_ptr<Volume> volume;
	std::shared_ptr<RayCastingParameters> rcp;
	
	VolumeCUDA volumeGradient;
	VolumeCUDA filteredVolumeGradient;

	float computeVectorEntropy(float* ary, int size);

	void GPU_setVolume(const VolumeCUDA *vol);
	void GPU_setConstants(float* _transFuncP1, float* _transFuncP2, float* _la, float* _ld, float* _ls, float3* _spacing);

	std::vector<SpherePoint> sphereSamples;
	int numSphereSample;
	float* d_sphereSamples = 0;

	const int nbins = 32;
	float* d_hist;
	std::vector<float*> cubeFaceHists;

	std::shared_ptr<Volume> resVol = 0;

	float * d_r = 0;

	void initJS06Sphere();
	void initTao09Detail();
	void initLabelVisibility();

	bool JS06SphereInited = false;
	bool Tao09DetailInited = false;
	bool LabelVisibilityInited = false;

	float computeLocalSphereEntropy(float3 eyeInLocal, VPMethod m);

	bool labelBeenSet = false;

	float3 indToLocal(int i, int j, int k);
	bool spherePointSet = false;
};





#endif