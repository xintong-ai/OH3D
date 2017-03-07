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

texture<float, 3, cudaReadModeElementType>  channelVolumeTexInput;
surface<void, cudaSurfaceType3D>			channelVolumeSurfaceOut;

__global__ void
d_updateVolumebyMatrixInfo_circluar(cudaExtent volumeSize, float3 start, float3 end, float3 spacing, float r) //channelVolumeSurfaceOut not changed yet
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
	{
		return;
	}

	float3 pos = make_float3(x, y, z) * spacing;
	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);
	float l = dot(pos - start, tunnelVec);
	if (l > 0 && l < tunnelLength){
		float disToStart = length(pos - start);
		float dis = sqrt(disToStart*disToStart - l*l);
		if (dis < r / 2){
			float res = 0;
			surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
		}
		else if (dis < r){
			float3 prjPoint = start + l*tunnelVec;
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

__global__ void
d_updateVolumebyMatrixInfo_rect(cudaExtent volumeSize, float3 start, float3 end, float3 spacing, float r, float deformationScale2nd, float3 dir2nd)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
	{
		return;
	}

	float3 pos = make_float3(x, y, z) * spacing;
	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);

	float3 voxelVec = pos - start;
	float l = dot(voxelVec, tunnelVec);
	if (l > 0 && l < tunnelLength){
		float disToStart = length(voxelVec);
		float l2 = dot(voxelVec, dir2nd);
		if (abs(l2) < deformationScale2nd){
			float3 prjPoint = start + l*tunnelVec + l2*dir2nd;
			float3 dir = normalize(pos - prjPoint);
			float dis = length(pos - prjPoint);
			float3 samplePos = prjPoint + dir*(r - (r - dis) * 2);

			if (dis < r / 2){
				float res = 0;
				surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);

				//float res2 = 1;
				//surf3Dwrite(res2, channelVolumeSurfaceOut, x * sizeof(float), y, z);
			}
			else if (dis < r){
				float3 prjPoint = start + l*tunnelVec + l2*dir2nd;
				float3 dir = normalize(start - prjPoint);
				float3 samplePos = prjPoint + dir*(r - (r - dis) * 2);

				float res = tex3D(volumeTexInput, samplePos.x + 0.5, samplePos.y + 0.5, samplePos.z + 0.5);
				surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);

				//float res2 = tex3D(channelVolumeTexInput, samplePos.x + 0.5, samplePos.y + 0.5, samplePos.z + 0.5);
				//surf3Dwrite(res2, channelVolumeSurfaceOut, x * sizeof(float), y, z);
			}
			else{
				float res = tex3D(volumeTexInput, x + 0.5, y + 0.5, z + 0.5);
				surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);

				//float res2 = tex3D(channelVolumeTexInput, x + 0.5, y + 0.5, z + 0.5);
				//surf3Dwrite(res2, channelVolumeSurfaceOut, x * sizeof(float), y, z);
			}
		}
		else{
			float res = tex3D(volumeTexInput, x + 0.5, y + 0.5, z + 0.5);
			surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);

			//float res2 = tex3D(channelVolumeTexInput, x + 0.5, y + 0.5, z + 0.5);
			//surf3Dwrite(res2, channelVolumeSurfaceOut, x * sizeof(float), y, z);
		}
	}
	else{
		float res = tex3D(volumeTexInput, x + 0.5, y + 0.5, z + 0.5);
		surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);

		//float res2 = tex3D(channelVolumeTexInput, x + 0.5, y + 0.5, z + 0.5);
		//surf3Dwrite(res2, channelVolumeSurfaceOut, x * sizeof(float), y, z);
	}
	return;
}

__global__ void
d_updateVolumebyMatrixInfo_tunnel_rect(cudaExtent volumeSize, float3 start, float3 end, float3 spacing, float r, float deformationScale2nd, float3 dir2nd)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
	{
		return;
	}

	float3 pos = make_float3(x, y, z) * spacing;
	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);

	float3 voxelVec = pos - start;
	float l = dot(voxelVec, tunnelVec);
	if (l > 0 && l < tunnelLength){
		float disToStart = length(voxelVec);
		float l2 = dot(voxelVec, dir2nd);
		if (abs(l2) < deformationScale2nd){
			float3 prjPoint = start + l*tunnelVec + l2*dir2nd;
			float3 dir = normalize(pos - prjPoint);
			float dis = length(pos - prjPoint);
			float3 samplePos = prjPoint + dir*(r - (r - dis) * 2);

			if (dis < r / 2){
				float res2 = 1;
				surf3Dwrite(res2, channelVolumeSurfaceOut, x * sizeof(float), y, z);
			}
			else if (dis < r){
				float3 prjPoint = start + l*tunnelVec + l2*dir2nd;
				float3 dir = normalize(start - prjPoint);
				float3 samplePos = prjPoint + dir*(r - (r - dis) * 2);

				float res2 = tex3D(channelVolumeTexInput, samplePos.x + 0.5, samplePos.y + 0.5, samplePos.z + 0.5);
				surf3Dwrite(res2, channelVolumeSurfaceOut, x * sizeof(float), y, z);
			}
			else{
				float res2 = tex3D(channelVolumeTexInput, x + 0.5, y + 0.5, z + 0.5);
				surf3Dwrite(res2, channelVolumeSurfaceOut, x * sizeof(float), y, z);
			}
		}
		else{
			float res2 = tex3D(channelVolumeTexInput, x + 0.5, y + 0.5, z + 0.5);
			surf3Dwrite(res2, channelVolumeSurfaceOut, x * sizeof(float), y, z);
		}
	}
	else{
		float res2 = tex3D(channelVolumeTexInput, x + 0.5, y + 0.5, z + 0.5);
		surf3Dwrite(res2, channelVolumeSurfaceOut, x * sizeof(float), y, z);
	}
	return;
}

__global__ void
d_posInDeformedChannelVolume(float3 pos, int3 dims, float3 spacing, bool* inChannel)
{
	float3 ind = pos / spacing;
	if (ind.x >= 0 && ind.x < dims.x && ind.y >= 0 && ind.y < dims.y && ind.z >= 0 && ind.z<dims.z) {
		float res = tex3D(channelVolumeTexInput, ind.x + 0.5, ind.y + 0.5, ind.z + 0.5); //?+0.5
		if (res > 0.5)
			*inChannel = true;
		else
			*inChannel = false;
	}
	else{
		*inChannel = true;
	}
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

	//cudaChannelFormatDesc cd2 = channelVolume->volumeCuda.channelDesc;
	//checkCudaErrors(cudaBindTextureToArray(channelVolumeTexInput, channelVolume->volumeCudaOri.content, cd2));
	//checkCudaErrors(cudaBindSurfaceToArray(channelVolumeSurfaceOut, channelVolume->volumeCuda.content));

	//d_updateVolumebyMatrixInfo_circluar << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, degree);

	d_updateVolumebyMatrixInfo_rect << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, degree, deformationScale2nd, rectDeformDir2nd);
	checkCudaErrors(cudaUnbindTexture(volumeTexInput));
	//checkCudaErrors(cudaUnbindTexture(channelVolumeTexInput));
}


void PositionBasedDeformProcessor::doTunnelDeforme(float degree)
{
	cudaExtent size = volume->volumeCuda.size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp(size.width, blockSize.x), iDivUp(size.height, blockSize.y), iDivUp(size.depth, blockSize.z));

	cudaChannelFormatDesc cd2 = channelVolume->volumeCuda.channelDesc;
	checkCudaErrors(cudaBindTextureToArray(channelVolumeTexInput, channelVolume->volumeCudaOri.content, cd2));
	checkCudaErrors(cudaBindSurfaceToArray(channelVolumeSurfaceOut, channelVolume->volumeCuda.content));

	//d_updateVolumebyMatrixInfo_circluar << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, degree);

	d_updateVolumebyMatrixInfo_tunnel_rect << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, deformationScale, deformationScale2nd, rectDeformDir2nd);
	checkCudaErrors(cudaUnbindTexture(channelVolumeTexInput));
}


void PositionBasedDeformProcessor::computeTunnelInfo()
{
	float3 eyeInLocal = matrixMgr->getEyeInLocal();
	float3 tunnelAxis = normalize(matrixMgr->recentMove);
	
	float step = 0.5;
	tunnelStart = eyeInLocal;
	tunnelEnd = eyeInLocal + tunnelAxis*step;
	while (channelVolume->inRange(tunnelEnd) && channelVolume->getVoxel(tunnelEnd) < 0.5){
		tunnelEnd += tunnelAxis*step;
	}

	float3 targetUpVecInLocal = make_float3(0, 0, 1);	//note! the vector make_float3(0,0,1) may also be used in ImmersiveInteractor class

	//rectDeformDir2nd = targetUpVecInLocal;
	if (abs(dot(targetUpVecInLocal, tunnelAxis)) < 0.9){
		rectDeformDir2nd = normalize(cross(cross(tunnelAxis, targetUpVecInLocal), tunnelAxis));
	}
	else{
		rectDeformDir2nd = matrixMgr->getViewVecInLocal();
	}
	//std::cout << "rectDeformDir2nd: " << rectDeformDir2nd.x << " " << rectDeformDir2nd.y << " " << rectDeformDir2nd.z << std::endl;
}

bool PositionBasedDeformProcessor::process(float* modelview, float* projection, int winWidth, int winHeight)
{
	if (!isActive)
		return false;

	float3 eyeInLocal = matrixMgr->getEyeInLocal();
	if (!hasBeenDeformed){
		if (volume->inRange(eyeInLocal) && channelVolume->getVoxel(eyeInLocal) < 0.5){
			// in solid area
			// in this case, set the start of deformation
			if (!hasBeenDeformed){
				hasBeenDeformed = true;
				
				computeTunnelInfo();

				doTunnelDeforme(deformationScale);

				//start a opening animation
				hasOpenAnimeStarted = true;
				hasCloseAnimeStarted = false;
				startOpen = std::clock();

				//currently if there is other tunnels exist, they are closed suddenly, because doing animation needs another group of tunnel info
			}
		}
		else{ 
			// either eyeInLocal is out of range, or eyeInLocal is in channel
			//in this case, do nothing, except close the tunnel which is currently opening
			if (hasCloseAnimeStarted){
				float r;
				double past = (std::clock() - startClose) / (double)CLOCKS_PER_SEC;
				if (past >= totalDuration){
					volume->reset();
					hasCloseAnimeStarted = false;
				}
				else{
					r = (1 - past / totalDuration)*closeStartingRadius;
					doDeforme(r);
				}
			}
		}
	}
	else{ //hasBeenDeformed

		if (volume->inRange(eyeInLocal) && channelVolume->getVoxel(eyeInLocal) < 0.5){
			//in area which is solid in the original volume

			bool* d_inchannel;
			cudaMalloc(&d_inchannel, sizeof(bool)* 1);
			cudaChannelFormatDesc cd2 = channelVolume->volumeCudaOri.channelDesc;
			checkCudaErrors(cudaBindTextureToArray(channelVolumeTexInput, channelVolume->volumeCuda.content, cd2));
			d_posInDeformedChannelVolume << <1, 1 >> >(eyeInLocal, volume->size, volume->spacing, d_inchannel);
			bool inchannel;
			cudaMemcpy(&inchannel, d_inchannel, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
			if (inchannel){
				// not in the solid region in the deformed volume
				// in this case, stable. set the start of deformation

				if (hasOpenAnimeStarted){
					float r;
					double past = (std::clock() - startOpen) / (double)CLOCKS_PER_SEC;
					if (past >= totalDuration){
						r = deformationScale;
						hasOpenAnimeStarted = false;
					}
					else{
						r = past / totalDuration*deformationScale;
					}
					closeStartingRadius = r;
					doDeforme(r);
				}
			}
			else{
				// even in the deformed volume, eye is still inside the solid region 

				//volume->reset();
				//channelVolume->reset();
				computeTunnelInfo();
				doTunnelDeforme(deformationScale);

				//start a opening animation
				hasOpenAnimeStarted = true;
				startOpen = std::clock();
			}
		}

		else{ 
			// in area which is channel in the original volume
			
			if (hasBeenDeformed){
				hasBeenDeformed = false;
				hasCloseAnimeStarted = true;
				hasOpenAnimeStarted = false;
				startClose = std::clock();
				channelVolume->reset();
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

	channelVolumeTexInput.normalized = false;
	channelVolumeTexInput.filterMode = cudaFilterModePoint;
	channelVolumeTexInput.addressMode[0] = cudaAddressModeBorder;
	channelVolumeTexInput.addressMode[1] = cudaAddressModeBorder;
	channelVolumeTexInput.addressMode[2] = cudaAddressModeBorder;
}

