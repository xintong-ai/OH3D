#pragma once
#include <ParticleReader.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <vector>
class timestep;
class SolutionParticleReader:public ParticleReader
{
	timestep* ts;
public:
	SolutionParticleReader(const char* filename) :ParticleReader(filename){ Load(); }
protected:
	void Load() override;
};
