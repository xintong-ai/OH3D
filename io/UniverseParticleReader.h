#pragma once
#include <ParticleReader.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <vector>
class UniverseParticleReader :public ParticleReader
{
public:
	UniverseParticleReader(const char* filename) :ParticleReader(filename){ Load(); }
protected:
	void Load() override;
};
