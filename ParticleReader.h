#ifndef PARTICLE_READER_H
#define PARTICLE_READER_H
#include <Reader.h>
//#include <fstream>
//#include <iostream>
//#define _USE_MATH_DEFINES
//#include "math.h"
#include <vector_types.h>

class ParticleReader:public Reader
{
	int num;
	float3* pos;
public:
	ParticleReader(const char* filename) :Reader(filename){ Load(); }
protected:
	void Load() override;
};



#endif