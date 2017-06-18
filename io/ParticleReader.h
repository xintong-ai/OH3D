#ifndef PARTICLE_READER_H
#define PARTICLE_READER_H
#include <vector_types.h>
#include <vector_functions.h>
#include <vector>
class timestep;
class ParticleReader
{
	timestep* ts;
public:
	ParticleReader(const char* filename) {
		datafilename.assign(filename);
	}

protected:
	virtual void Load() = 0;
	std::vector<float> val;
	std::vector<float4> pos;
	std::vector<char> feature; //actually segmentation label
	int num;
	std::string datafilename;

};



#endif