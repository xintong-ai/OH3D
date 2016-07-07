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
class FlightReader :public Reader
{
	int num;
	//float4* pos;
	timestep* ts;
	std::vector<float> val;
public:
	FlightReader(const char* filename) :Reader(filename){ Load(); }
	//float4* GetPos();
	std::vector<float4> GetPos();
	std::vector<float4> pos;
	int GetNum();
	//float* GetVal();
	std::vector<float> GetVal();
	void GetValRange(float& vMin, float& vMax);
	void GetPosRange(float3& posMin, float3& posMax) override;
protected:
	void Load() override;
};



#endif