#include <Volume.h>
#include <iostream>

#include <fstream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

void VolumeCUDA::VolumeCUDA_init(int3 _size, float *volumeVoxelValues, int allowStore, int numChannels)
{
	/// !!! note!!! only correct when VolumeType is float !!!

	//if (!volumeVoxelValues){
	//	cout << "VolumeCUDA_init input value invalid" << endl;
	//	exit(0);
	//}

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
	if (volumeVoxelValues){
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(volumeVoxelValues, size.width*sizeof(VolumeType)* numChannels, size.width, size.height);
		copyParams.dstArray = content;
		copyParams.extent = size;
		copyParams.kind = cudaMemcpyHostToDevice;
		checkCudaErrors(cudaMemcpy3D(&copyParams));
	}
}

void VolumeCUDA::VolumeCUDA_init(int3 _size, unsigned short *volumeVoxelValues, int allowStore, int numChannels)
{
	size = make_cudaExtent(_size.x, _size.y, _size.z);
	if (numChannels == 4)
	{
		channelDesc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);
	}
	else if (numChannels == 1)
	{
		channelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
	}
	else
	{
		channelDesc = cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsigned);
	}

	checkCudaErrors(cudaMalloc3DArray(&content, &channelDesc, size, allowStore ? cudaArraySurfaceLoadStore : 0));

	// copy data to 3D array
	if (volumeVoxelValues){
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(volumeVoxelValues, size.width*sizeof(unsigned short)* numChannels, size.width, size.height);
		copyParams.dstArray = content;
		copyParams.extent = size;
		copyParams.kind = cudaMemcpyHostToDevice;
		checkCudaErrors(cudaMemcpy3D(&copyParams));
	}
	else{
		unsigned short *temp = new unsigned short[size.width*size.height*size.depth];
		memset(temp, 0, sizeof(unsigned short)*size.width*size.height*size.depth);

		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(temp, size.width*sizeof(unsigned short)* numChannels, size.width, size.height);
		copyParams.dstArray = content;
		copyParams.extent = size;
		copyParams.kind = cudaMemcpyHostToDevice;
		checkCudaErrors(cudaMemcpy3D(&copyParams));
		delete[] temp ;
	}
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

void Volume::initVolumeCuda(){
	volumeCuda.VolumeCUDA_deinit();
	volumeCuda.VolumeCUDA_init(size, values, 1, 1); //the third parameter means allowStore or not
	volumeCudaOri.VolumeCUDA_deinit();
	volumeCudaOri.VolumeCUDA_init(size, values, 0, 1);
}

void Volume::reset(){
	volumeCuda.VolumeCUDA_deinit();
	volumeCuda.VolumeCUDA_init(size, values, 1, 1);
}

void Volume::saveRawToFile(const char *f)
{
	FILE * fp = fopen(f, "wb");
	fwrite(values, sizeof(float), size.x*size.y*size.z, fp);
	fclose(fp);
}
