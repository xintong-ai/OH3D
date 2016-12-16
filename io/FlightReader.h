#ifndef PARTICLE_READER_H
#define PARTICLE_READER_H

//#include <fstream>
//#include <iostream>
//#define _USE_MATH_DEFINES
//#include "math.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <vector>
class timestep;

class FlightReader
{
	int num;
	//float4* pos;
	timestep* ts;
	std::vector<float> val;
	void TranslateToCenter();

public:
	FlightReader(const char* filename){ Load(filename); }
	//float4* GetPos();
	std::vector<float4> GetPos();
	std::vector<float4> pos;
	int GetNum();
	//float* GetVal();
	std::vector<float> GetVal();
	void GetValRange(float& vMin, float& vMax);
	void GetPosRange(float3& posMin, float3& posMax);
protected:
	void Load(const char* filename);
};



#endif