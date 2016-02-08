#include "VecReader.h"
#include <vector>
#include <iostream>
//#include <fstream>
//#include <cstdint>
//#include <vector_functions.h>
#include <helper_math.h>


void VecReader::Load()
{
	FILE *pFile;
	pFile = fopen(datafilename.c_str(), "rb");
	if (pFile == NULL) { fputs("File error", stderr); exit(1); }
	fread(size, sizeof(int), 3, pFile);
	num = size[0] * size[1] * size[2];
	float* voxelValues = new float[3 * num];
	vecs = new float3[num];
	fread(voxelValues, sizeof(float), 3 * num, pFile);
	valMin = 99999;
	valMax = -1;
	float meaninglessThr = 10000;
	for (int i = 0; i < num; i++){
		if (abs(voxelValues[3 * i])>meaninglessThr || abs(voxelValues[3 * i + 1]) > meaninglessThr || abs(voxelValues[3 * i + 2]) > meaninglessThr){
			vecs[i] = make_float3(0, 0, 0);
			val.push_back(0);
		}
		else{
			vecs[i] = make_float3(voxelValues[3 * i], voxelValues[3 * i + 1], voxelValues[3 * i + 2]);
			float value = length(vecs[i]);
			val.push_back(value);
			if (value > valMax)
				valMax = value;
			if (value < valMin)
				valMin = value;
		}
			
	}
	delete voxelValues;
	fclose(pFile);
	
	//std::cout << "num : " << num << std::endl;
	//for (int i = 0; i < 30; i++){
	//	std::cout << "fiber: " << vecs[i].x << " " << vecs[i].y << " " << vecs[i] .z << std::endl;
	//}
}

void VecReader::GetPosRange(float3& posMin, float3& posMax)
{
	posMin = make_float3(0.0, 0.0, 0.0);
	posMax = make_float3(size[0] - 1, size[1] - 1, size[2] - 1);
}

void VecReader::GetSamples(std::vector<float4>& _pos, std::vector<float3>& _vec, std::vector<float>& _val)
{

	////for isabel
	//float lengthThrLow = 40;
	//float lengthThrHigh = 1000;
	//for (int k = 0; k < size[2]; k += 4){
	//	for (int j = 2 + (k / 4) % 4; j < 250; j += 8) {
	//		for (int i = 200+(j / 4) % 4; i < 400; i += 8){


	//for plume
	float lengthThrLow = 0.2;
	float lengthThrHigh = 1;
	for (int k = size[2]*6/10; k < size[2]*9/10; k+=4){
		for (int j = 0; j < size[1]; j+=4) {
			for (int i = 0; i < size[0]; i += 4){
				int idx = k * size[0] * size[1] + j * size[0] + i;
				if (val[idx] > lengthThrLow && val[idx] <lengthThrHigh) {
					_pos.push_back(make_float4(i, j, k, 1.0));
					_val.push_back(val[idx]);
					_vec.push_back(vecs[idx]);
				}
			}
		}
	}
}
