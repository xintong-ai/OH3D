#include <Volume.h>
#include <iostream>

#include <fstream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

void VolumeCUDA::VolumeCUDA_init(int3 _size, float *volumeVoxelValues, int allowStore, int numChannels)
{
	VolumeCUDA_deinit();

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
	VolumeCUDA_deinit();

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

void VolumeCUDA::VolumeCUDA_contentUpdate(unsigned short *volumeVoxelValues, int allowStore, int numChannels)
{
	if (content == 0){
		std::cout << "error!!!!!" << std::endl;
	}

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
		delete[] temp;
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
	if (originSaved){
		volumeCudaOri.VolumeCUDA_deinit();
		volumeCudaOri.VolumeCUDA_init(size, values, 0, 1);
	}
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

void Volume::computeGradient()
{
	if (gradient != 0) delete[] gradient;

	gradient = new float[size.x*size.y*size.z * 4];

	for (int z = 0; z < size.z; z++){
		for (int y = 0; y < size.y; y++){
			for (int x = 0; x < size.x; x++){
				int ind = (z*size.y*size.x + y*size.x + x) * 4;

				int indz1 = z - 2, indz2 = z + 2;
				if (indz1 < 0)	indz1 = 0;
				if (indz2 > size.z - 1) indz2 = size.z - 1;
				gradient[ind + 2] = (values[indz2*size.y*size.x + y*size.x + x] - values[indz1*size.y*size.x + y*size.x + x]) / (indz2 - indz1);

				int indy1 = y - 2, indy2 = y + 2;
				if (indy1 < 0)	indy1 = 0;
				if (indy2 > size.y - 1) indy2 = size.y - 1;
				gradient[ind + 1] = (values[z*size.y*size.x + indy2*size.x + x] - values[z*size.y*size.x + indy1*size.x + x]) / (indy2 - indy1);

				int indx1 = x - 2, indx2 = x + 2;
				if (indx1 < 0)	indx1 = 0;
				if (indx2 > size.x - 1) indx2 = size.x - 1;
				gradient[ind] = (values[z*size.y*size.x + y*size.x + indx2] - values[z*size.y*size.x + y*size.x + indx1]) / (indx2 - indx1);

				gradient[ind + 3] = 0;

				float l = sqrt(gradient[ind] * gradient[ind] + gradient[ind + 1] * gradient[ind + 1] + gradient[ind + 2] * gradient[ind + 2]);
				if (l > maxGadientLength)
					maxGadientLength = l;
			}
		}
	}
}

void Volume::computeGradient(float* &f)
{
	//note!! this function stores the gradient in float4 tuples, because it is preparing to copy the data to cudaArray, which does not support float3 well

	f = new float[size.x*size.y*size.z*4];
	
	for (int z= 0; z < size.z; z++){
		for (int y = 0; y < size.y; y++){
			for (int x = 0; x < size.x; x++){
				int ind = (z*size.y*size.x + y*size.x + x) * 4;

				int indz1 = z - 2, indz2 = z + 2;
				if (indz1 < 0)	indz1 = 0;
				if (indz2 > size.z - 1) indz2 = size.z - 1;
				f[ind + 2] = (values[indz2*size.y*size.x + y*size.x + x] - values[indz1*size.y*size.x + y*size.x + x]) / (indz2 - indz1);

				int indy1 = y - 2, indy2 = y + 2;
				if (indy1 < 0)	indy1 = 0;
				if (indy2 > size.y - 1) indy2 = size.y - 1;
				f[ind + 1] = (values[z*size.y*size.x + indy2*size.x + x] - values[z*size.y*size.x + indy1*size.x + x]) / (indy2 - indy1);

				int indx1 = x - 2, indx2 = x + 2;
				if (indx1 < 0)	indx1 = 0;
				if (indx2 > size.x - 1) indx2 = size.x - 1;
				f[ind] = (values[z*size.y*size.x + y*size.x + indx2] - values[z*size.y*size.x + y*size.x + indx1]) / (indx2 - indx1);

				f[ind + 3] = 0;
			}
		}
	}
}

void Volume::computeGradient(float* input, int3 size, float* &f)
{
	//note!! this function stores the gradient in float4 tuples, because it is preparing to copy the data to cudaArray, which does not support float3 well

	f = new float[size.x*size.y*size.z * 4];

	for (int z = 0; z < size.z; z++){
		for (int y = 0; y < size.y; y++){
			for (int x = 0; x < size.x; x++){
				int ind = (z*size.y*size.x + y*size.x + x) * 4;

				int indz1 = z - 2, indz2 = z + 2;
				if (indz1 < 0)	indz1 = 0;
				if (indz2 > size.z - 1) indz2 = size.z - 1;
				f[ind + 2] = (input[indz2*size.y*size.x + y*size.x + x] - input[indz1*size.y*size.x + y*size.x + x]) / (indz2 - indz1);

				int indy1 = y - 2, indy2 = y + 2;
				if (indy1 < 0)	indy1 = 0;
				if (indy2 > size.y - 1) indy2 = size.y - 1;
				f[ind + 1] = (input[z*size.y*size.x + indy2*size.x + x] - input[z*size.y*size.x + indy1*size.x + x]) / (indy2 - indy1);

				int indx1 = x - 2, indx2 = x + 2;
				if (indx1 < 0)	indx1 = 0;
				if (indx2 > size.x - 1) indx2 = size.x - 1;
				f[ind] = (input[z*size.y*size.x + y*size.x + indx2] - input[z*size.y*size.x + y*size.x + indx1]) / (indx2 - indx1);

				f[ind + 3] = 0;
			}
		}
	}
}


void Volume::computeBilateralFiltering(float* &res, float sigs, float sigr)
{
	res = new float[size.x*size.y*size.z];

	for (int z = 0; z < size.z; z++){
		for (int y = 0; y < size.y; y++){
			for (int x = 0; x < size.x; x++){
				int ind = z*size.y*size.x + y*size.x + x;
				
				float IP = values[ind];

				double sum = 0, sumwp = 0;				
				//float sum = 0, sumwp = 0;

				for (int zz = -1; zz <= 1; zz++){
					for (int yy = -1; yy <= 1; yy++){
						for (int xx = -1; xx <= 1; xx++){
							int xq = x + xx, yq = y + yy, zq = z + zz;
							if (xq >= 0 && xq < size.x && yq >= 0 && yq < size.y && zq >= 0 && zq < size.z) {
								int ind2 = zq*size.y*size.x + yq*size.x + xq;
								float IQ = values[ind2];

								double gq = exp(-(xx*xx + yy*yy + zz*zz) / sigs)*exp(-(IP - IQ) / sigr);
								//float gq = exp(-(xx*xx + yy*yy + zz*zz) / sigs - (IP - IQ) / sigr);
								sumwp += gq;
								sum += gq*IQ;
							}
						}
					}
				}

				if (sumwp > 0)
					res[ind] = sum / sumwp;
				else
					res[ind] = 0;
			}
		}
	}
}