#ifndef VEC_READER_H
#define VEC_READER_H
//#include <fstream>
//#include <iostream>
//#define _USE_MATH_DEFINES
//#include "math.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <vector>
#include <memory>
#include <Volume.h>


class VecReader 
{
	int num;
	int size[3];
	float3 *vecs;

	//float4* pos;
	//timestep* ts;

	std::vector<float> val;
public:
	VecReader(const char* filename) { Load(filename); }
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
	
	void GetPosRange(float3& posMin, float3& posMax) ;
	void GetSamples(std::vector<float4>& _pos, std::vector<float3>& _vec, std::vector<float>& _val);

	void OutputToVolumeByNormalizedVecMag(std::shared_ptr<Volume> v);
	void OutputToVolumeByNormalizedVecDownSample(std::shared_ptr<Volume> v, int c);
	void OutputToVolumeByNormalizedVecUpSample(std::shared_ptr<Volume> v, int c);
	void OutputToVolumeByNormalizedVecMagWithPadding(std::shared_ptr<Volume> v, int nn);

protected:
	void Load(const char* filename);
};



#endif