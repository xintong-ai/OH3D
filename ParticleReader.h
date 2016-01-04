#ifndef PARTICLE_READER_H
#define PARTICLE_READER_H
#include <Reader.h>
//#include <fstream>
//#include <iostream>
//#define _USE_MATH_DEFINES
//#include "math.h"
#include <vector_types.h>
class timestep;
class ParticleReader:public Reader
{
	int num;
	float3* pos;
	timestep* ts;
public:
	ParticleReader(const char* filename) :Reader(filename){ Load(); }
	float3* GetPos();
	int GetNum();
	float* GetConcentration();
	void GetDataRange(float3& posMin, float3& posMax);
protected:
	void Load() override;
};



#endif