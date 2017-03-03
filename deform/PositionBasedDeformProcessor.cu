#include "PositionBasedDeformProcessor.h"
#include "Lens.h"
#include "MeshDeformProcessor.h"
#include "Volume.h"
#include "TransformFunc.h"
#include "MatrixManager.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

texture<float, 3, cudaReadModeElementType>  volumeTexInput;
surface<void, cudaSurfaceType3D>			volumeSurfaceOut;

__global__ void
d_updateVolumebyMatrixInfo(cudaExtent volumeSize, float3 start, float3 end, float3 spacing, float r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
	{
		return;
	}

	float3 pos = make_float3(x, y, z) * spacing;
	float3 tunnelVec = end - start;
	float3 viewVec = normalize(tunnelVec);
	float tunnelLength = length(tunnelVec);
	float l = dot(pos - start, viewVec);
	if (l > 0 && l < tunnelLength){
		float disToStart = length(pos - start);
		float dis = sqrt(disToStart*disToStart - l*l);
		if (dis < r / 2){
			float res = 0;
			surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
		}
		else if (dis < r){
			float3 prjPoint = start + l*viewVec;
			float3 dir = normalize(start - prjPoint);
			float3 samplePos = prjPoint + dir*(r - (r - dis) * 2);

			float res = tex3D(volumeTexInput, samplePos.x + 0.5, samplePos.y + 0.5, samplePos.z + 0.5);
			surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
		}
		else{
			float res = tex3D(volumeTexInput, x + 0.5, y + 0.5, z + 0.5);
			surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
		}
	}
	else{
		float res = tex3D(volumeTexInput, x + 0.5, y + 0.5, z + 0.5);
		surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
	}
	return;
}

void PositionBasedDeformProcessor::doDeforme(float degree)
{
	cudaExtent size = volume->volumeCuda.size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp(size.width, blockSize.x), iDivUp(size.height, blockSize.y), iDivUp(size.depth, blockSize.z));

	cudaChannelFormatDesc cd = volume->volumeCudaOri.channelDesc;
	checkCudaErrors(cudaBindTextureToArray(volumeTexInput, volume->volumeCudaOri.content, cd));

	checkCudaErrors(cudaBindSurfaceToArray(volumeSurfaceOut, volume->volumeCuda.content));
	d_updateVolumebyMatrixInfo << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, degree);

	checkCudaErrors(cudaUnbindTexture(volumeTexInput));
}

bool PositionBasedDeformProcessor::process(float* modelview, float* projection, int winWidth, int winHeight)
{
	if (!isActive)
		return false;

	float3 eyeInLocal = matrixMgr->getEyeInLocal();
	if (volume->inRange(eyeInLocal)){
		float v = channelVolume->getVoxel(eyeInLocal);
		//if (v < 0.5){
		//	std::cout << "checked" << std::endl;
		//}
	
		if (v < 0.5){
			if (!hasDeformed){
				hasDeformed = true;
				hasAnimeStarted = true;
				start = std::clock();

				float3 tunnelAxis = matrixMgr->getHorizontalMoveVec(make_float3(0,0,1));
				//note! the vector make_float3(0,0,1) may also be used in ImmersiveInteractor class
				//std::cout << "viewVecInLocal: " << moveVecInLocal.x << " " << moveVecInLocal.y << " " << moveVecInLocal.z << std::endl;
				float step = 0.5;
				tunnelStart = eyeInLocal;
				tunnelEnd = eyeInLocal + tunnelAxis*step;
				while (channelVolume->inRange(tunnelEnd) && channelVolume->getVoxel(tunnelEnd) < 0.5){
					tunnelEnd += tunnelAxis*step;
				}
			}
			if (hasAnimeStarted){
				
				float r;
				double past = (std::clock() - start) / (double)CLOCKS_PER_SEC;
				if (past >= totalDuration){
					r = maxRadius;
					hasAnimeStarted = false;
				}
				else{
					r = past / totalDuration*maxRadius;
				}
				closeStartingRadius = r;

				doDeforme(r);
			}
			
		}
		else{
			//if (hasDeformed){
			//	hasDeformed = false;
			//	hasAnimeStarted = false;
			//	volume->reset();
			//}

			if (hasDeformed){
				hasDeformed = false;
				hasAnimeStarted = true;
				start = std::clock();
			}
			if (hasAnimeStarted){
				float r;
				double past = (std::clock() - start) / (double)CLOCKS_PER_SEC;
				if (past >= totalDuration){
					volume->reset();
					hasAnimeStarted = false;
				}
				else{
					r = (1 - past / totalDuration)*closeStartingRadius;
				
					doDeforme(r);
				}

			}
		}
	}
	return false;
}

void PositionBasedDeformProcessor::InitCudaSupplies()
{
	volumeTexInput.normalized = false;
	volumeTexInput.filterMode = cudaFilterModeLinear;
	volumeTexInput.addressMode[0] = cudaAddressModeBorder;
	volumeTexInput.addressMode[1] = cudaAddressModeBorder;
	volumeTexInput.addressMode[2] = cudaAddressModeBorder;
}

