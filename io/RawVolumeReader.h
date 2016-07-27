#ifndef RAWVOLUME_READER_H
#define RAWVOLUME_READER_H
#include "vector_types.h"
#include "Reader.h"
#include <assert.h>

#include <algorithm>
#include <helper_math.h>
#include <vector>
#include <string>

#include <Volume.h>

class RawVolumeReader :public Reader
{
public:
	void Load() override;

	RawVolumeReader(const char* filename, int3 _dim); //need to provide the dimensions at least
	RawVolumeReader(const char* filename, int3 _dim, float3 _origin, float3 _spacing, bool _isUnsignedShort);

	void OutputToVolumeByNormalizedValue(std::shared_ptr<Volume> v);

	//void GetDataOrigin(float3& val) { val = dataOrigin; }

	//void GetDataSize(int val[3]) {
	//	val[0] = dataSizes.x;
	//	val[1] = dataSizes.y;
	//	val[2] = dataSizes.z;
	//}

	//int GetDataSizeMin() { return std::min(std::min(dataSizes.x, dataSizes.y), dataSizes.z); }

	//int3 GetDataSize(){ return dataSizes; }

	//float3 GetDataMin(){ return dataMin; }

	//float3 GetDataMax(){ return dataMax; }

	////void GetValRange(float& vMin, float& vMax) override;

	
	//int GetDataSize(int i) { assert(i < 3); return *(&dataSizes.x + i); }

	//float GetMaxVal();

	//float GetDataOrigin(int i) { assert(i < 3); return (&dataOrigin.x)[i]; }

	//float3 GetDataSpaceDir(int i) { assert(i < 3); return dataSpaceDir[i]; }

	//void GetDataRange(float& min, float& max){ min = minVal; max = maxVal; }

	//void* GetData() { return (void *)data; }

	//float3 GetDataPos(int3 p);


	void GetPosRange(float3& posMin, float3& posMax) override;

	~RawVolumeReader();

protected:

	int3 dataSizes;

	bool isUnsignedShort = true; //currently support unsigned short and float only
	
	float3 dataOrigin = make_float3(0,0,0);
	float3 spacing = make_float3(1,1,1);

	float maxVal = 0.0f;
	float minVal = 0.0f;
	float *data = nullptr;
};

#endif