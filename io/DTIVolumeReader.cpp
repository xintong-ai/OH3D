#include "DTIVolumeReader.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>
#include  "eig3.h"

#define EPS 1e-6

DTIVolumeReader::DTIVolumeReader(const char* filename) 
	:VolumeReader(filename)
{
}

void DTIVolumeReader::EigenAnalysis()
{
	//if it is already computed, return
	if (nullptr != eigenvec)
		return;
	int nCells = dataSizes.x * dataSizes.y * dataSizes.z;
	eigenvec = new float[9 * nCells];
	eigenval = new float[3 * nCells];

	for (int i = 0; i < nCells; i++) {
		eigen_decomposition<float>((float*)data + 9 * i, eigenvec + 9 * i, eigenval + 3 * i);
	}
}

float3* DTIVolumeReader::GetMajorComponent()
{
	if (nullptr != majorEigenvec)
		return majorEigenvec;
	EigenAnalysis();

	int nCells = dataSizes.x * dataSizes.y * dataSizes.z;
	majorEigenvec = new float3[nCells];
	for (int i = 0; i < nCells; i++) {
		majorEigenvec[i] = make_float3(eigenvec[i * 9], eigenvec[i * 9 + 1], eigenvec[i * 9 + 2]);
	}
	return majorEigenvec;
}

float3* DTIVolumeReader::GetColors()
{
	if (nullptr != colors)
		return colors;
	const float scale = 1.0;
	const float3 white = 0.2 * make_float3(1,1,1);
	float3* a = GetMajorComponent();
	float* b = GetFractionalAnisotropy();
	int nCells = dataSizes.x * dataSizes.y * dataSizes.z;
	colors = new float3[nCells];
	for (int i = 0; i < nCells; i++) {
		colors[i] = a[i] * b[i];
		colors[i] = make_float3(
			abs(colors[i].z) * scale,
			abs(colors[i].y) * scale,
			abs(colors[i].x) * scale)
			+white;
	}
	return colors;
}


//float* DTIVolumeReader::GetEigenValue()
//{
//	EigenAnalysis();
//}

float* DTIVolumeReader::GetFractionalAnisotropy()
{
	if (nullptr != fracAnis)
		return fracAnis;
	EigenAnalysis();
	int nCells = dataSizes.x * dataSizes.y * dataSizes.z;
	fracAnis = new float[nCells];
	float v1, v2, v3, v_avg, v1_v, v2_v, v3_v, det;
	float c_3 = 1.0f / 3.0f;
	float c_2_3 = sqrt(3.0f / 2.0f);
	for (int i = 0; i < nCells; i++) {
		v1 = eigenval[i * 3];
		v2 = eigenval[i * 3 + 1];
		v3 = eigenval[i * 3 + 2];
		v_avg = (v1 + v2 + v3) * c_3;
		v1_v = v1 - v_avg;
		v2_v = v2 - v_avg;
		v3_v = v3 - v_avg;
		det = v1 * v1 + v2 * v2 + v3 * v3;
		if ( det < EPS)
			fracAnis[i] = 0;
		else
			fracAnis[i] = c_2_3 
			* sqrt(v1_v * v1_v + v2_v * v2_v + v3_v * v3_v)
			/ sqrt(det);
	}
	return fracAnis;
}

inline float4 float3To4(float3& v)
{
	return make_float4(v.x, v.y, v.z, 1.0f);
}

void DTIVolumeReader::GetSamples(std::vector<float4>& _pos, std::vector<float>& _val)
{
	float* b = GetFractionalAnisotropy();
	int nCells = dataSizes.x * dataSizes.y * dataSizes.z;

	for (int k = 0; k < dataSizes.z; k++){
		for (int j = 0; j < dataSizes.y; j+=4) {
			for (int i = 0; i < dataSizes.x; i+=4){
				int idx = k * dataSizes.x * dataSizes.y + j * dataSizes.x + i;
				if (b[idx] > 0.4) {
					_pos.push_back(float3To4(GetDataPos(make_int3(i, j, k))));
					_val.push_back(1.0f);
					_val.push_back(*((float*)data + 9 * idx));
					_val.push_back(*((float*)data + 9 * idx + 1));
					_val.push_back(*((float*)data + 9 * idx + 2));
					_val.push_back(*((float*)data + 9 * idx + 4));
					_val.push_back(*((float*)data + 9 * idx + 5));
					_val.push_back(*((float*)data + 9 * idx + 8));
				}
			}
		}
	}
}

void DTIVolumeReader::GetSamplesWithFeature(std::vector<float4>& _pos, std::vector<float>& _val, std::vector<char> &_feature)
{
	float* b = GetFractionalAnisotropy();
	int nCells = dataSizes.x * dataSizes.y * dataSizes.z;

	for (int k = 0; k < dataSizes.z; k++){
		for (int j = 0; j < dataSizes.y; j += 4) {
			for (int i = 0; i < dataSizes.x; i += 4){
				int idx = k * dataSizes.x * dataSizes.y + j * dataSizes.x + i;
				if (b[idx] > 0.3 ){//|| (feature[idx]>0 && b[idx] > 0.2)) {
					_pos.push_back(float3To4(GetDataPos(make_int3(i, j, k))));
					_feature.push_back(feature[idx]);
					_val.push_back(1.0f);
					_val.push_back(*((float*)data + 9 * idx));
					_val.push_back(*((float*)data + 9 * idx + 1));
					_val.push_back(*((float*)data + 9 * idx + 2));
					_val.push_back(*((float*)data + 9 * idx + 4));
					_val.push_back(*((float*)data + 9 * idx + 5));
					_val.push_back(*((float*)data + 9 * idx + 8));
				}
			}
		}
	}
}