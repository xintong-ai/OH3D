#include <Volume.h>
#include <iostream>

#include <fstream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

void VolumeCUDA::VolumeCUDA_init(int3 _size, float *volumeVoxelValues, int allowStore, int numChannels)
{
	/// !!! note!!! only correct when VolumeType is float !!!

	if (!volumeVoxelValues){
		cout << "VolumeCUDA_init input value invalid" << endl;
		exit(0);
	}

	size = make_cudaExtent(_size.x, _size.y, _size.z);
	if (numChannels == 4)
	{
		channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	}
	else if (numChannels == 1)
	{
		channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	}
	else
	{
		channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
	}


	checkCudaErrors(cudaMalloc3DArray(&content, &channelDesc, size, allowStore ? cudaArraySurfaceLoadStore : 0));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(volumeVoxelValues, size.width*sizeof(VolumeType)* numChannels, size.width, size.height);
	copyParams.dstArray = content;
	copyParams.extent = size;
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

}

VolumeCUDA::~VolumeCUDA()
{
	VolumeCUDA_deinit();
}

void VolumeCUDA::VolumeCUDA_deinit()
{
	if (content != 0)
		cudaFreeArray(content);
	content = 0;
}



void Volume::GetPosRange(float3& posMin, float3& posMax)
{
	posMin = dataOrigin;
	posMax = dataOrigin + spacing*make_float3(size);
}

void Volume::initVolumeCuda(int allowStore){
	volumeCuda.VolumeCUDA_deinit();
	volumeCuda.VolumeCUDA_init(size, values, allowStore, 1);
}