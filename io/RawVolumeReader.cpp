#include "RawVolumeReader.h"
#include "string.h"
#include <iostream>

//#include "cuda_math.h"
#include <vector_functions.h>

//for linux
#include <helper_math.h>
//using namespace std;
#include <float.h>

RawVolumeReader::RawVolumeReader(const char* filename, int3 _dim) :
Reader(filename)
{
	dataSizes = _dim;
	isUnsignedShort = true;
	dataOrigin = make_float3(0, 0, 0);
	spacing = make_float3(1, 1, 1);
		
	Load();
}

RawVolumeReader::RawVolumeReader(const char* filename, int3 _dim, float3 _origin, float3 _spacing, bool _isUnsignedShort) :
Reader(filename)
{
	dataSizes = _dim;
	isUnsignedShort = _isUnsignedShort;
	dataOrigin = _origin;
	spacing = _spacing;

	Load();
}

void RawVolumeReader::Load()
{
	if (nullptr != data) {
		delete[] data;
	}

	data = new float[dataSizes.x*dataSizes.y*dataSizes.z];

	FILE *pFile;
	pFile = fopen(datafilename.c_str(), "rb");
	if (pFile == NULL) { fputs("volume file error", stderr); exit(1); }

	if (!isUnsignedShort) //raw data format is float
	{
		fread(data, sizeof(float), dataSizes.x*dataSizes.y*dataSizes.z, pFile);
	}
	else //raw data format is unsigned short
	{
		unsigned short *tempread = new unsigned short[dataSizes.x*dataSizes.y*dataSizes.z];
		fread(tempread, sizeof(unsigned short), dataSizes.x*dataSizes.y*dataSizes.z, pFile);
		for (int k = 0; k < dataSizes.z; k++)
		{
			for (int j = 0; j < dataSizes.y; j++)
			{
				for (int i = 0; i < dataSizes.x; i++)
				{
					float v = tempread[k*dataSizes.y * dataSizes.x + j*dataSizes.x + i];
					if (v>maxVal)
						maxVal = v;
					if (v < minVal)
						minVal = v;

					data[k*dataSizes.y * dataSizes.x + j*dataSizes.x + i] = v;
				}
			}
		}
		delete tempread;
	}
	fclose(pFile);

	std::cout << "before normalization, volume value range " << minVal << " " << maxVal << std::endl;


}


RawVolumeReader::~RawVolumeReader()
{
	if (nullptr != data) {
		delete[] data;
	}
}

void RawVolumeReader::GetPosRange(float3& posMin, float3& posMax)
{
	posMin = dataOrigin;
	posMax = dataOrigin + spacing*make_float3(dataSizes);
}


void RawVolumeReader::OutputToVolumeByNormalizedValue(std::shared_ptr<Volume> v)
{
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
				v->values[ind] = (data[ind] - minVal) / (maxVal - minVal);
			}
		}
	}

}
