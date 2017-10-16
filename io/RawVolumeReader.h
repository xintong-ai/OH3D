#ifndef RAWVOLUME_READER_H
#define RAWVOLUME_READER_H
#include <vector_types.h>
#include <assert.h>

#include <algorithm>
#include <helper_math.h>
#include <vector>
#include <string>

#include "Volume.h"

#ifndef _DataType_
#define _DataType_
struct DataType {
	bool isFloat;
	bool isSigned;
	int bitsPerSample;

	bool operator ==(DataType a){
		return isFloat == a.isFloat && isSigned == a.isSigned && bitsPerSample == a.bitsPerSample;
	}
};
#endif

class RawVolumeReader 
{
public:
	///	define 7 common types of data type
	static const DataType dtFloat32, dtInt8, dtUint8, dtInt16, dtUint16, dtInt32, dtUint32;

	RawVolumeReader(const char* filename, int3 _dim, DataType _DataType = dtUint16); //need to provide the dimensions at least
	~RawVolumeReader();

	void OutputToVolumeByNormalizedValue(std::shared_ptr<Volume> v);
	void OutputToVolumeCUDAUnsignedShort(std::shared_ptr<VolumeCUDA>);
	void OutputToVolume_OnlyVolumeCuda_NoNormalized(std::shared_ptr<Volume> v);

	static void rawFileReadingInfo(std::string dataPath, DataType & volDataType, bool &labelFromFile)
	{
		if (std::string(dataPath).find("knee") != std::string::npos || std::string(dataPath).find("engine") != std::string::npos || std::string(dataPath).find("181") != std::string::npos || std::string(dataPath).find("Bucky") != std::string::npos || std::string(dataPath).find("bloodCell") != std::string::npos || std::string(dataPath).find("Lobster") != std::string::npos || std::string(dataPath).find("Orange") != std::string::npos || std::string(dataPath).find("Tomato") != std::string::npos || std::string(dataPath).find("Neghip") != std::string::npos)
			volDataType = dtUint8;
		else
			volDataType = dtUint16;

		if (std::string(dataPath).find("Baseline") != std::string::npos)
			labelFromFile = true;
	};

protected:
	std::string datafilename;
	int3 dataSizes;

	bool isUnsignedShort = true; //currently support unsigned short and float only
	
	float3 dataOrigin = make_float3(0,0,0);
	float3 spacing = make_float3(1,1,1);

	double minVal, maxVal;

	DataType m_DataType = dtUint16;
	void* m_Data = 0;

	void GetMinMaxValue();
	void Clean();
	void Allocate();

	void Load();

};

#endif