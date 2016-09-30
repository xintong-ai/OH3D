#include "VecReader.h"
#include <vector>
#include <iostream>
#include <fstream>
//#include <cstdint>
//#include <vector_functions.h>
#include <helper_math.h>
#include <cmath>

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


	////for plume
	//float lengthThrLow = 0.1;
	//float lengthThrHigh = 1.2;
	//for (int k = size[2] * 0.4; k < size[2] * 0.6; k += 4){
	//	for (int j = 0; j < size[1]; j+=4) {
	//		for (int i = 0; i < size[0]; i += 4){
	//			int idx = k * size[0] * size[1] + j * size[0] + i;
	//			if (val[idx] > lengthThrLow && val[idx] <lengthThrHigh) {
	//				_pos.push_back(make_float4(i, j, k, 1.0));
	//				_val.push_back(val[idx]);
	//				_vec.push_back(vecs[idx]);
	//			}
	//		}
	//	}
	//}

	//for nek
	float lengthThrLow = 20;
	float lengthThrHigh = 1000003.2;
	for (int k = 0; k < size[2]; k += 4){
		for (int j = 0; j < size[1]; j += 4) {
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



void VecReader::OutputToVolumeByNormalizedVecMagWithPadding(std::shared_ptr<Volume> v, int nn){

	v->~Volume();

	v->size = make_int3(size[0], size[1], size[2]) + make_int3(0, nn, 0);;

	v->spacing = make_float3(1.0, 1.0, 1.0); //may add spacing info into the vecReader?
	v->dataOrigin = make_float3(0, 0, 0);

	v->values = new float[v->size.x*v->size.y*v->size.z];
	for (int k = 0; k < v->size.z; k++)
	{
		for (int j = 0; j < v->size.y; j++)
		{
			for (int i = 0; i < v->size.x; i++)
			{
				int ind = k*v->size.y * v->size.x + j*v->size.x + i;
				int jj = j - nn;
				if (jj < 0){
					v->values[ind] = 0.0;
				}
				else{
					int indOri = k*size[1] * size[0] + jj*size[0] + i;
					v->values[ind] = (val[indOri] - valMin) / (valMax - valMin);
				}
			}
		}
	}
}
void VecReader::OutputToVolumeByNormalizedVecMag(std::shared_ptr<Volume> v){
	v->~Volume();

	v->size = make_int3(size[0], size[1], size[2]);

	v->spacing = make_float3(1.0, 1.0, 1.0); //may add spacing info into the vecReader?
	v->dataOrigin = make_float3(0, 0, 0);

	v->values = new float[v->size.x*v->size.y*v->size.z];
	for (int k = 0; k < v->size.z; k++)
	{
		for (int j = 0; j < v->size.y; j++)
		{
			for (int i = 0; i < v->size.x; i++)
			{
				int ind = k*v->size.y * v->size.x + j*v->size.x + i;
				v->values[ind] = (val[ind] - valMin) / (valMax - valMin);
			}
		}
	}

	//std::ofstream OutFile;
	//OutFile.open("nek256mag", std::ofstream::out | std::ofstream::binary);
	//OutFile.write((char*)v->values, sizeof(float)*v->size.x*v->size.y*v->size.z);
	//OutFile.close();
}



void VecReader::OutputToVolumeByNormalizedVecDownSample(std::shared_ptr<Volume> v, int n)
{
	//float n = 1.5;

	v->~Volume();

	v->size = make_int3(size[0] / n, size[1] / n, size[2] / n);

	v->spacing = make_float3(1.0 * n, 1.0 * n, 1.0 * n);
	v->dataOrigin = make_float3(0, 0, 0);

	v->values = new float[v->size.x*v->size.y*v->size.z];
	for (int k = 0; k < v->size.z; k++)
	{
		for (int j = 0; j < v->size.y; j++)
		{
			for (int i = 0; i < v->size.x; i++)
			{
				int ind = k*v->size.y * v->size.x + j*v->size.x + i;
				int indOri = int(k*n)*size[0] * size[1] + int(j*n)*size[0] + int(i*n);
				v->values[ind] = (val[indOri] - valMin) / (valMax - valMin);
			}
		}
	}
}


void VecReader::OutputToVolumeByNormalizedVecUpSample(std::shared_ptr<Volume> v, int n)
{
	//float n = 1.5;

	v->~Volume();

	v->size = make_int3(size[0] * n, size[1] * n, size[2] * n);

	v->spacing = make_float3(1.0 / n, 1.0 / n, 1.0 / n);
	v->dataOrigin = make_float3(0, 0, 0);

	v->values = new float[v->size.x*v->size.y*v->size.z];
	for (int k = 0; k < v->size.z; k++)
	{
		for (int j = 0; j < v->size.y; j++)
		{
			for (int i = 0; i < v->size.x; i++)
			{
				int ind = k*v->size.y * v->size.x + j*v->size.x + i;
				int indOri = k/n*size[0] * size[1] + j/n*size[0] + i/n;
				//int indOri = k / 3 * 2 * size[0] * size[1] + j / 3 * 2 * size[0] + i / 3 * 2;
				v->values[ind] = (val[indOri] - valMin) / (valMax - valMin);
			}
		}
	}
}