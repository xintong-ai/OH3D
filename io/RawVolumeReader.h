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
	void OutputToVolumeByNormalizedValueWithPadding(std::shared_ptr<Volume> v, int nn);

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