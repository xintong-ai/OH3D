#ifndef VIEWPOINTEVALUATOR_H
#define VIEWPOINTEVALUATOR_H

#define _USE_MATH_DEFINES
#include <cmath>

#include "myDefine.h"
#include <memory>
#include <vector>


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
	ViewpointEvaluator(std::shared_ptr<Volume> v);
	~ViewpointEvaluator(){};

	std::shared_ptr<Volume> volume;
	RayCastingParameters rcp;

	float computeVolumewhiseEntropy(float3 eyeInWorld, float * d_r);
	float computeSpherewhiseEntropy(float3 eyeInWorld, float * d_r);


	void setSpherePoints(int n);
	std::vector<SpherePoint> sphereSamples;
	int numSphereSample = 0;


private:
	void GPU_setVolume(const VolumeCUDA *vol);
	void GPU_setConstants(float* _transFuncP1, float* _transFuncP2, float* _la, float* _ld, float* _ls, float3* _spacing);

	const int nbins = 20;
	float* d_hist;
	float* d_sphereSamples = 0;

};





#endif