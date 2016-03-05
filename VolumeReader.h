#ifndef VOLUME_READER_H
#define VOLUME_READER_H
#include "vector_types.h"
#include "Reader.h"
#include <assert.h>

#include <algorithm>
#include <helper_math.h>


struct matrix3x3{
	float3 v[3];
};

inline float3 operator*(matrix3x3 a, float3 b)
{
	float3 ret;
	ret.x = dot(a.v[0], b);
	ret.y = dot(a.v[1], b);
	ret.z = dot(a.v[2], b);
	return ret;
}

class VolumeReader:public Reader
{
public:
	void LoadNRRD(const char* filename);
	void Load() override;

	VolumeReader(const char* filename);
	
	int GetNumFields() { return numFields; }
	
	void GetDataOrigin(float3& val) {val = dataOrigin;}

//	void GetSlice(int dir, int sliceNum, float* data, int nx, int ny);
	void GetSlice(int sliceDirIdx, int sliceNum, int fieldIdx, float*& out, int& size1, int& size2);

	
	void GetDataSpaceDir(float3 val[3]) {
		val[0] = dataSpaceDir[0];
		val[1] = dataSpaceDir[1];
		val[2] = dataSpaceDir[2];
	}


	void GetDataSize(int val[3]) {
		val[0] = dataSizes.x;
		val[1] = dataSizes.y;
		val[2] = dataSizes.z;
	}

	int GetDataSizeMin() { return std::min(std::min(dataSizes.x, dataSizes.y), dataSizes.z); }

	int3 GetDataSize(){ return dataSizes; }

	float3 GetDataMin(){ return dataMin; }

	float3 GetDataMax(){ return dataMax; }

	//void GetValRange(float& vMin, float& vMax) override;

	void GetPosRange(float3& posMin, float3& posMax) override;

	int GetDataSize(int i) { assert(i < 3); return *(&dataSizes.x + i); }

	float GetMaxVal();

	float GetDataOrigin(int i) { assert(i < 3); return (&dataOrigin.x)[i]; }

	float3 GetDataSpaceDir(int i) { assert(i < 3); return dataSpaceDir[i]; }

	void GetDataRange(float& min, float& max){min = minVal; max = maxVal;}

	void* GetData() {return (void *)data;}

	float3 GetDataPos(int3 p);

	~VolumeReader();

	bool useFeature;
	void loadFeature(const char* filename)
	{
		FILE *pFile;
		pFile = fopen(filename, "rb");
		if (pFile == NULL) { fputs("File error", stderr); exit(1); }
		int num = dataSizes.x*dataSizes.y*dataSizes.z;
		feature = new char[num];
		short *temp = new short[num];
		fread(temp, sizeof(char), num, pFile);
		for (int i = 0; i < num; i++){
			feature[i] = temp[i];
		}
	};

protected:
	int dataDims;
	int3 dataSizes;
	float3 dataOrigin;
	float3 dataSpaceDir[3];
	int numFields = 1;
	float3 dataMin, dataMax;

	float maxVal = 0.0f;
	float minVal = 0.0f;
	float *data = nullptr;

	char *feature = nullptr;

};

#endif