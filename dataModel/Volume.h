#ifndef VOLUME_H
#define VOLUME_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <memory>
//using namespace std;

#include <cuda_runtime.h>
#include <helper_cuda.h>

typedef float VolumeType;

class VolumeCUDA
{
public:
	cudaExtent size;
	cudaArray *content = 0;
	cudaChannelFormatDesc channelDesc;
	
	void VolumeCUDA_init(int3 _size, float *volumeVoxelValues, int allowStore, int numChannels = 1);
	void VolumeCUDA_init(int3 _size, unsigned short *volumeVoxelValues, int allowStore, int numChannels = 1);
	void VolumeCUDA_contentUpdate(unsigned short *volumeVoxelValues, int allowStore, int numChannels = 1);


	~VolumeCUDA();

	void VolumeCUDA_deinit();
};



class Volume
{
public:

	int3 size;
	
	float *values = 0; //used to store the voxel values. generally normalized to [0,1] if used for volume rendering

	//float getVoxel(int3 v){
	//	return values[v.z*size.x*size.y + v.y*size.x + v.x];
	//};

	float getVoxel(float3 v){
		return values[int(v.z)*size.x*size.y + int(v.y)*size.x + int(v.x)];
	};
	bool inRange(float3 v){
		return v.x >= 0 && v.x < size.x && v.y >= 0 && v.y < size.y &&v.z >= 0 && v.z < size.z;
	}
	void setSize(int3 s){ 
		size = s;
		if (values != 0){
			delete values;
		}
		values = new float[size.x*size.y*size.z];
		memset(values, 0, sizeof(float)*size.x*size.y*size.z);
	};

	Volume(bool _so = false){ originSaved = _so; };
	//Volume(){ };

	~Volume()
	{
		if (!values) delete values;
	};


	float3 spacing = make_float3(1.0,1.0,1.0);
	float3 dataOrigin = make_float3(0, 0, 0);
	void GetPosRange(float3& posMin, float3& posMax);

	bool originSaved;
	VolumeCUDA volumeCudaOri;
	VolumeCUDA volumeCuda; //using two copies, since the volumeCuda might be deformed

	void initVolumeCuda();
	void reset();

	void saveRawToFile(const char *);

};
#endif