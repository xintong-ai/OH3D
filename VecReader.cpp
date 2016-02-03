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
	for (int i = 0; i < num; i++){
		vecs[i] = make_float3(voxelValues[3 * i], voxelValues[3 * i + 1], voxelValues[3 * i + 2]);
		float value = length(vecs[i]);
		val.push_back(value);
		if (value > valMax)
			valMax = value;
		if (value < valMin)
			valMin = value;
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

