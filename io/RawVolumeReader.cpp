#include "RawVolumeReader.h"
#include <string.h>
#include <iostream>

//#include "cuda_math.h"
#include <vector_functions.h>

//for linux
#include <helper_math.h>
//using namespace std;


const DataType RawVolumeReader::dtFloat32 = { true, true, 32 };
const DataType RawVolumeReader::dtInt8 = { false, true, 8 };
const DataType RawVolumeReader::dtUint8 = { false, false, 8 };
const DataType RawVolumeReader::dtInt16 = { false, true, 16 };
const DataType RawVolumeReader::dtUint16 = { false, false, 16 };
const DataType RawVolumeReader::dtInt32 = { false, true, 32 };
const DataType RawVolumeReader::dtUint32 = { false, false, 32 };

RawVolumeReader::RawVolumeReader(const char* filename, int3 _dim)
{
	datafilename.assign(filename);
	dataSizes = _dim;
	
	dataOrigin = make_float3(0, 0, 0);
	spacing = make_float3(1, 1, 1);
		
	Load();

	GetMinMaxValue();
}

void RawVolumeReader::Clean()
{
	if (m_Data)
	{
		free(m_Data);
		m_Data = 0;
	}
}

void RawVolumeReader::Allocate()
{
	Clean();
	m_Data = malloc((unsigned long long)(m_DataType.bitsPerSample >> 3)*(unsigned long long)dataSizes.x * (unsigned long long)dataSizes.y * (unsigned long long)dataSizes.z);
	if (!m_Data) {
		printf("malloc fails...\n");
	}
}

template<typename T>
void t_GetMinMaxValue(T* data, unsigned size, double &minVal, double &maxVal)
{
	unsigned i;
	T min_v, max_v;
	min_v = max_v = *data;

	T* pData = data + 1;

	for (i = 1; i<size; i++, pData++)
	{
		if (min_v>*pData) min_v = *pData;
		if (max_v<*pData) max_v = *pData;
	}

	minVal = min_v;
	maxVal = max_v;
}

void RawVolumeReader::GetMinMaxValue()
{
	unsigned long long size = (unsigned long long)dataSizes.x * (unsigned long long)dataSizes.y * (unsigned long long)dataSizes.z;
	if (m_DataType.isFloat) t_GetMinMaxValue((float*)m_Data, size, minVal, maxVal);
	else if (m_DataType.isSigned)
	{
		if (m_DataType.bitsPerSample == 8) t_GetMinMaxValue((char*)m_Data, size, minVal, maxVal);
		else if (m_DataType.bitsPerSample == 16) t_GetMinMaxValue((short*)m_Data, size, minVal, maxVal);
		else if (m_DataType.bitsPerSample == 32) t_GetMinMaxValue((int*)m_Data, size, minVal, maxVal);
	}
	else
	{
		if (m_DataType.bitsPerSample == 8) t_GetMinMaxValue((unsigned char*)m_Data, size, minVal, maxVal);
		else if (m_DataType.bitsPerSample == 16) t_GetMinMaxValue((unsigned short*)m_Data, size, minVal, maxVal);
		else if (m_DataType.bitsPerSample == 32) t_GetMinMaxValue((unsigned int*)m_Data, size, minVal, maxVal);
	}
}



void RawVolumeReader::Load()
{
	FILE * fp;
#ifdef _LINUX_86_64
	fp = fopen(filename, "rb");
#else
	fopen_s(&fp, datafilename.c_str(), "rb");
#endif
	printf("%s\n", datafilename.c_str());
	Allocate();
	fread(m_Data, m_DataType.bitsPerSample >> 3, (unsigned long long)dataSizes.x * (unsigned long long)dataSizes.y * (unsigned long long)dataSizes.z, fp);
	fclose(fp);
	//printf("%dx%dx%d\n", m_Dims[0], m_Dims[1], m_Dims[2]);
}


RawVolumeReader::~RawVolumeReader()
{
	Clean();
}


void RawVolumeReader::OutputToVolumeByNormalizedValue(std::shared_ptr<Volume> v)
{
	if (m_DataType == dtUint16){
		v->~Volume();

		v->size = dataSizes;

		v->spacing = spacing;
		v->dataOrigin = dataOrigin;

		v->values = new float[dataSizes.x*dataSizes.y*dataSizes.z];

		for (int k = 0; k < dataSizes.z; k++)
		{
			for (int j = 0; j < dataSizes.y; j++)
			{
				for (int i = 0; i < dataSizes.x; i++)
				{
					int ind = k*dataSizes.y * dataSizes.x + j*dataSizes.x + i;
					v->values[ind] = (((unsigned short*)m_Data)[ind] - minVal) / (maxVal - minVal);
				}
			}
		}
	}
	else{
		std::cout << "not implement" << std::endl;
		exit(0);
	}
}
