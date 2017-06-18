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
	std::vector<float> val; //currently majorly used for magnitude

	float valMin, valMax; //used to compute normalized magnitude

public:
	VecReader(const char* filename) { Load(filename); }


	void GetSamples(std::vector<float4>& _pos, std::vector<float3>& _vec, std::vector<float>& _val); //used to compute components of Particle class, by select vectors whose magnitude within two thresholds. need future work to connect to a Particle object

	void OutputToVolumeByNormalizedVecMag(std::shared_ptr<Volume> v);
	void OutputToVolumeByNormalizedVecDownSample(std::shared_ptr<Volume> v, int c);
	void OutputToVolumeByNormalizedVecUpSample(std::shared_ptr<Volume> v, int c);
	void OutputToVolumeByNormalizedVecMagWithPadding(std::shared_ptr<Volume> v, int nn);

protected:
	void Load(const char* filename);
};



#endif