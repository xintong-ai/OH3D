#ifndef PARTICLE_READER_H
#define PARTICLE_READER_H
#include <Reader.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <vector>
class timestep;
class ParticleReader:public Reader
{
	//float4* pos;
	timestep* ts;
public:
	ParticleReader(const char* filename) :Reader(filename){}
	//float4* GetPos();
	std::vector<float4> GetPos();
	int GetNum();
	//float* GetVal();
	std::vector<float> GetVal();
	void GetValRange(float& vMin, float& vMax);
	void GetPosRange(float3& posMin, float3& posMax) override;
protected:
	virtual void Load() = 0;
	std::vector<float> val;
	std::vector<float4> pos;
	std::vector<char> feature; //actually segmentation label
	int num;
};



#endif