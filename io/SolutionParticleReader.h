#pragma once
#include <ParticleReader.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <vector>
#include <Particle.h>

class timestep;
class SolutionParticleReader:public ParticleReader
{
	timestep* ts;
public:
	//SolutionParticleReader(const char* filename) :ParticleReader(filename){ Load(); }
	SolutionParticleReader(const char* filename, float _thr) :ParticleReader(filename){
		thr = _thr; 
		Load(); 
	}
	float thr;
	void OutputToParticleData(std::shared_ptr<Particle> v);

protected:
	void Load() override;
};
