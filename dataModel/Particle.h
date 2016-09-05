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
	std::vector<float4> pos; //current position after deformation
	int numParticles;

	std::vector<float> val; //attribute that might be important. currently only support one attribute
	float valMin, valMax;

	bool hasFeature = false;
	std::vector<char> feature; //actually is segmentation tag. should be named as "label" or "tag"
	void setFeature(std::vector<char> _f){
		feature = _f;
		hasFeature = true;
	};


	Particle(){};
	Particle(std::vector<float4> _pos, std::vector<float> _val);
	
	~Particle()
	{
	};

private:

};
#endif