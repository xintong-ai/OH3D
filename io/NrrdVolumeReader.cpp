#include "NrrdVolumeReader.h"
#include "string.h"

#include "NRRD/nrrd_image.hxx"
//#include "cuda_math.h"
#include <vector_functions.h>

//for linux
#include <helper_math.h>
//using namespace std;
#include <float.h>

NrrdVolumeReader::NrrdVolumeReader(const char* filename)
{
	datafilename.assign(filename);
	Load();
}

void NrrdVolumeReader::Load()
{
	LoadNRRD(datafilename.c_str());
}


void NrrdVolumeReader::GetSlice(int sliceDirIdx, int sliceNum, int fieldIdx, float*& out, int& size1, int& size2)
{
	int size0 = *(&dataSizes.x + sliceDirIdx);
	size1 = *(&dataSizes.x + (sliceDirIdx + 1) % 3);
	size2 = *(&dataSizes.x + (sliceDirIdx + 2) % 3);
	out = new float[size1 * size2];

	switch (sliceDirIdx)
	{
	case 0:
		for (int j = 0; j < size2; j++) {
			for (int i = 0; i < size1; i++) {
				int idx = j * size1 * size0 + i * size0 + sliceNum;
				float a = *(data + idx * numFields + fieldIdx);
				out[j * size1 + i] = a;
			}
		}
		break;
	case 1:
		for (int j = 0; j < size2; j++) {
			for (int i = 0; i < size1; i++) {
				int idx = i * size2 * size0 + sliceNum * size2 + j;
				out[j * size1 + i] = *(data + idx * numFields);
			}
		}
		break;
	case 2:
		for (int j = 0; j < size2; j++) {
			for (int i = 0; i < size1; i++) {
				int idx = size1 * size2 * sliceNum + j * size1 + i;
				out[j * size1 + i] = *(data + idx * numFields + fieldIdx);
			}
		}
		break;
	}

}
float3 NrrdVolumeReader::GetDataPos(int3 p)
{
	matrix3x3 spaceDirMat; 
	spaceDirMat.v[0] = dataSpaceDir[0];
	spaceDirMat.v[1] = dataSpaceDir[1];
	spaceDirMat.v[2] = dataSpaceDir[2];
	return spaceDirMat * make_float3(p) + dataOrigin;
}


void NrrdVolumeReader::LoadNRRD(const char* filename)
{
	std::string file = filename;
	NRRD::Image<float> img(file);
	if (!img) {
		std::cerr << "Failed to read file.nrrd.\n";
		exit(1);
	}

	dataDims = img.dimension();
	if (4 == dataDims) {
		for (int i = 1; i < dataDims; i++) {
			*(&(dataSizes.x) + i - 1) = img.size(i);
			vector<double> tmp = img.space_directions(i);
			dataSpaceDir[i - 1] = make_float3(tmp[0], tmp[1], tmp[2]);
			(&dataOrigin.x)[i - 1] = img.space_origin(i - 1);
		}
		numFields = img.size(0);
	}
	else if (3 == dataDims) {
		for (int i = 0; i < dataDims; i++) {
			*(&(dataSizes.x) + i) = img.size(i);
			vector<double> tmp = img.space_directions(i);
			dataSpaceDir[i] = make_float3(tmp[0], tmp[1], tmp[2]);
			(&dataOrigin.x)[i] = img.space_origin(i);
		}
//		numFields = 1;
	}


	float3 dataEnd = dataOrigin
		+ dataSpaceDir[0] * (dataSizes.x - 1)
		+ dataSpaceDir[1] * (dataSizes.y - 1)
		+ dataSpaceDir[2] * (dataSizes.z - 1);

	// TODO: verify the correctness of the following lines
	//we may need to compare the 8 cube vertices
	for (int i = 0; i < 3; i++) {
		(&dataMin.x)[i] = min((&dataOrigin.x)[i], (&dataEnd.x)[i]);
		(&dataMax.x)[i] = max((&dataOrigin.x)[i], (&dataEnd.x)[i]);
	}


	int n = dataSizes.x * dataSizes.y * dataSizes.z * numFields;
	maxVal = -FLT_MAX;
	minVal = FLT_MAX;
	for (int i = 0; i < n; i++) {
		maxVal = img[i] > maxVal ? img[i] : maxVal;
		minVal = img[i] < minVal ? img[i] : minVal;
	}
	data = (float*)(&img[0]);

	// Supports N-dimensional images
	std::cout << "Number of dimensions: " << img.dimension() << std::endl;
	std::cout << "Size:";
	for (int i = 0; i<img.dimension(); i++)
		std::cout << " " << img.size(i);
	std::cout << std::endl;

	// Simple read/write access to image data (1D, 2D and 3D only)
	//img.pixel(0, 0, 0) = 1.0;
}

void NrrdVolumeReader::GetPosRange(float3& posMin, float3& posMax)
{
	posMin = dataMin;
	posMax = dataMax;
}


NrrdVolumeReader::~NrrdVolumeReader()
{
	if (nullptr != data) {
		delete[] data;
	}
}


bool NrrdVolumeReader::LoadFeature(const char* filename)
{
	int num = dataSizes.x*dataSizes.y*dataSizes.z;
	feature = new char[num];

	FILE *pFile;
	pFile = fopen(filename, "rb");
	if (pFile == NULL)
	{
		fputs("no feature is loaded \n", stderr);
		memset(feature, 0, num);
		return false;
	}

	typedef unsigned char FEATURE_FILE_TYPE;
	FEATURE_FILE_TYPE *temp = new FEATURE_FILE_TYPE[num];
	fread(temp, sizeof(FEATURE_FILE_TYPE), num, pFile);

	int featureAmount = 0;
	for (int i = 0; i < num; i++){
		feature[i] = temp[i];
		if (feature[i]>featureAmount)
			featureAmount = feature[i];
	}

	featureCenter.resize(featureAmount, make_float3(0, 0, 0));
	vector<int> featureCount(featureAmount,0);

	for (int k = 0; k < dataSizes.z; k++){
		for (int j = 0; j < dataSizes.y; j += 4) {
			for (int i = 0; i < dataSizes.x; i += 4){
				int idx = k * dataSizes.x * dataSizes.y + j * dataSizes.x + i;
				if (feature[idx]>0){
					featureCount[feature[idx] - 1] ++;
					featureCenter[feature[idx] - 1] += GetDataPos(make_int3(i, j, k));
				}
			}
		}
	}

	for (int i = 0; i < featureAmount; i++){
		featureCenter[i] /= featureCount[i];
	}

	return true;
}

bool NrrdVolumeReader::LoadFeatureNew(const char* filename)
{
	int num = dataSizes.x*dataSizes.y*dataSizes.z;
	feature = new char[num];

	FILE *pFile;
	pFile = fopen(filename, "rb");
	if (pFile == NULL)
	{
		fputs("no feature is loaded \n", stderr);
		memset(feature, 0, num);
		return false;
	}

	typedef unsigned char FEATURE_FILE_TYPE;
	FEATURE_FILE_TYPE *temp = new FEATURE_FILE_TYPE[num];
	fread(temp, sizeof(FEATURE_FILE_TYPE), num, pFile);

	int featureAmount = 0;
	for (int i = 0; i < num; i++){
		feature[i] = temp[i];
		if (feature[i]>featureAmount)
			featureAmount = feature[i];
	}
	featureAmount = (int)log2(featureAmount) + 1;

	featureCenter.resize(featureAmount, make_float3(0, 0, 0));
	vector<int> featureCount(featureAmount, 0);

	for (int k = 0; k < dataSizes.z; k++){
		for (int j = 0; j < dataSizes.y; j += 4) {
			for (int i = 0; i < dataSizes.x; i += 4){
				int idx = k * dataSizes.x * dataSizes.y + j * dataSizes.x + i;
				int v = feature[idx];
				for (int ff = 0; ff < featureAmount; ff++){
					if ((v / ((int)pow(2, ff))) % 2){
						featureCount[ff] ++;
						featureCenter[ff] += GetDataPos(make_int3(i, j, k));
					}
				}
			}
		}
	}

	for (int i = 0; i < featureAmount; i++){
		featureCenter[i] /= featureCount[i];
	}

	return true;
}

bool NrrdVolumeReader::LoadFeature2(std::vector<std::string> featureFiles)
{
	int featureAmount = featureFiles.size();
	int num = dataSizes.x*dataSizes.y*dataSizes.z;
	feature = new char[num];
	memset(feature, 0, num);

	for (int i = 0; i < featureAmount; i++){
		FILE *pFile;
		pFile = fopen(featureFiles[i].c_str(), "rb");
		if (pFile == NULL)
		{
			fputs("loading feature failure \n", stderr);
			memset(feature, 0, num);
			featureAmount = 0;
			return false;
		}

		typedef unsigned char FEATURE_FILE_TYPE;
		FEATURE_FILE_TYPE *temp = new FEATURE_FILE_TYPE[num];
		fread(temp, sizeof(FEATURE_FILE_TYPE), num, pFile);

		for (int i = 0; i < num; i++){
			if (temp[i]>0)
			feature[i] += pow(2,i);
		}
	}

	return true;
}
