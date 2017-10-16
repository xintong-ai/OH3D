#pragma once
#include <ParticleReader.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <vector>
#include <Particle.h>

//this should replace BinaryParticleReader in the future
class BinaryTuplesReader :public ParticleReader
{
	//timestep* ts;
public:
	BinaryTuplesReader(const char* filename) :ParticleReader(filename){ Load(); }

	void OutputToParticleData(std::shared_ptr<Particle> v);
	void OutputToParticleDataArrays(std::vector<std::shared_ptr<Particle>> & v);
protected:
	int numTupleArrays = 0;
	
	std::vector<std::vector<float>> valArrays;
	std::vector<std::vector<float4>> posArrays;
	std::vector<std::vector<char>> featureArrays;

	void Load() override;

private:
	void ReadArray(std::vector<float4> & curArray, std::vector<float> & curValArray, std::vector<char> & curFeatureArray, FILE *pFile);
};
