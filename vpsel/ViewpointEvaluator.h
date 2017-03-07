#ifndef VIEWPOINTEVALUATOR_H
#define VIEWPOINTEVALUATOR_H

#define _USE_MATH_DEFINES
#include <cmath>

#include "myDefine.h"
#include <memory>
#include <vector>
class Particle;

enum VPMethod{
	BS05,
	JS06Sphere
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
	};
	void setSpherePoints(int n = 512);

	void setLabel(std::shared_ptr<VolumeCUDA> labelVol);

	std::shared_ptr<Volume> volume;
	std::shared_ptr<RayCastingParameters> rcp;
	void initDownSampledResultVolume(int3 sampleSize);	
	void compute_UniformSampling(VPMethod m);
	void compute_SkelSampling(VPMethod m);
	void saveResultVol(const char*);

	float3 optimalEyeInLocal;
	bool optimalEyeValid = false; //when the optimalEye has not been computed, optimalEyeInLocal should not be used

	std::vector<float> cubeInfo;
	void computeCubeEntropy(float3 eyeInLocal, float3 viewDir, float3 upDir);
	
	std::vector<std::shared_ptr<Particle>> skelViews;
	
	float computeVectorEntropy(float* ary, int size);

	bool useHist = true, useTrad = false;
	bool useLabelCount = true, useColor = false;
	bool useDist = false;//not well set yet
	int maxLabel = 1; //!! data dependant

private:
	void GPU_setVolume(const VolumeCUDA *vol);
	void GPU_setConstants(float* _transFuncP1, float* _transFuncP2, float* _la, float* _ld, float* _ls, float3* _spacing);

	std::vector<SpherePoint> sphereSamples;
	int numSphereSample;
	float* d_sphereSamples = 0;



	const int nbins = 32;
	float* d_hist;
	std::vector<float*> cubeFaceDist;

	std::shared_ptr<Volume> resVol = 0;

	float * d_r = 0;

	void initBS05();
	void initJS06Sphere();
	bool BS05Inited = false;
	bool JS06SphereInited = false;
	float computeEntropyBS05(float3 eyeInLocal);
	float computeEntropyJS06Sphere(float3 eyeInLocal);

	bool labelBeenSet = false;

	float3 indToLocal(int i, int j, int k);

};





#endif