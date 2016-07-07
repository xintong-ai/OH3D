#ifndef VEC_READER_H
#define VEC_READER_H
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
class VecReader :public Reader
{
	int num;
	int size[3];
	float3 *vecs;

	//float4* pos;
	//timestep* ts;

	std::vector<float> val;
public:
	VecReader(const char* filename) :Reader(filename){ Load(); }
	//float4* GetPos();

	int GetNum(){
		return num;
	};

	std::vector<float> GetVal() {
		return val;
	};

	float valMin, valMax;
	void GetValRange(float& vMin, float& vMax){
		vMin = valMin, vMax = valMax;
	};
	
	void GetPosRange(float3& posMin, float3& posMax) override;
	void GetSamples(std::vector<float4>& _pos, std::vector<float3>& _vec, std::vector<float>& _val);

protected:
	void Load() override;
};



#endif