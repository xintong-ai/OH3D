#ifndef PARTICLE_READER_H
#define PARTICLE_READER_H
#include <Reader.h>
//#include <fstream>
//#include <iostream>
//#define _USE_MATH_DEFINES
//#include "math.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <vector>
class timestep;
//struct Particle{
//	float3 pos;
//	float val;
//	Particle(float x, float y, float z, float v) { pos = make_float3(x, y, z); val = v; }
//};
class ParticleReader:public Reader
{
	int num;
	//float4* pos;
	timestep* ts;
	std::vector<float> val;
public:
	ParticleReader(const char* filename) :Reader(filename){ Load(); }
	float4* GetPos();
	std::vector<float4> pos;
	int GetNum();
	float* GetVal();
	void GetDataRange(float3& posMin, float3& posMax);
protected:
	void Load() override;
};



#endif