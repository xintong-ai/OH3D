#include "PositionBasedDeformProcessor.h"
#include "Lens.h"
#include "MeshDeformProcessor.h"
#include "Volume.h"
#include "TransformFunc.h"
#include "MatrixManager.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>


//!!! NOTE !!! spacing not considered yet!!!! in the global functions


texture<float, 3, cudaReadModeElementType>  volumeTexInput;
surface<void, cudaSurfaceType3D>			volumeSurfaceOut;

texture<float, 3, cudaReadModeElementType>  channelVolumeTexInput;
surface<void, cudaSurfaceType3D>			channelVolumeSurfaceOut;


__device__ float3 sampleDis(float3 pos, float3 start, float3 end, float r, float deformationScaleVertical, float3 dir2nd)
{
	const float3 noChangeMark = make_float3(-1, -2, -3);
	const float3 emptyMark = make_float3(-3, -2, -1);

	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);

	float3 voxelVec = pos - start;
	float l = dot(voxelVec, tunnelVec);
	if (l > 0 && l < tunnelLength){
		float disToStart = length(voxelVec);
		float l2 = dot(voxelVec, dir2nd);
		if (abs(l2) < deformationScaleVertical){
			float3 prjPoint = start + l*tunnelVec + l2*dir2nd;
			float3 dir = normalize(pos - prjPoint);
			float dis = length(pos - prjPoint);
			float3 samplePos = prjPoint + dir*(r - (r - dis) * 2);

			if (dis < r / 2){
				return emptyMark;
			}
			else if (dis < r){
				float3 prjPoint = start + l*tunnelVec + l2*dir2nd;
				float3 dir = normalize(start - prjPoint);
				float3 samplePos = prjPoint + dir*(r - (r - dis) * 2);

				return samplePos;
			}
			else{
				return noChangeMark;
			}
		}
		else{
			return noChangeMark;
		}
	}
	else{
		return noChangeMark;
	}
}



__global__ void
d_updateVolumebyMatrixInfo_rect(cudaExtent volumeSize, float3 start, float3 end, float3 spacing, float r, float deformationScaleVertical, float3 dir2nd)
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
		if (abs(l2) < deformationScaleVertical){
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
				float3 samplePos = prjPoint + dir*(r - (r - dis) * 2); //!!! NOTE !!! spacing not considered yet!!!!

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
d_updateVolumebyMatrixInfo_rect_2anime(cudaExtent volumeSize, float3 spacing, float3 start, float3 end, float r, float deformationScaleVertical, float3 dir2nd, float lastDeformationDegree, float3 lastDeformationDirVertical, float3 lastTunnelStart, float3 lastTunnelEnd, float rClose)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
	{
		return;
	}

	float3 pos = make_float3(x, y, z) * spacing;

	float3 posOpen = sampleDis(pos, start, end, r, deformationScaleVertical, dir2nd);
	float3 posClose = sampleDis(pos, lastTunnelStart, lastTunnelEnd, rClose, deformationScaleVertical, lastDeformationDirVertical);

	if (posOpen.x < 0 && posOpen.z < posOpen.x && posClose.x < 0 && posClose.z < posClose.x){ //both no change of sample position
		float res = tex3D(volumeTexInput, x + 0.5, y + 0.5, z + 0.5);
		surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
	}
	else if (posOpen.x < 0 && posOpen.z < posOpen.x){//for open no change of sample position, just regular close
		if (posClose.x < 0 && posClose.x < posClose.z){
			float res = 0;
			surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
		}
		else{
			float res = tex3D(volumeTexInput, posClose.x + 0.5, posClose.y + 0.5, posClose.z + 0.5);
			surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
		}
	}
	else if (posClose.x < 0 && posClose.z < posClose.x){//for close no change of sample position, just regular open
		if (posOpen.x < 0 && posOpen.x < posOpen.z){
			float res = 0;
			surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
		}
		else{
			float res = tex3D(volumeTexInput, posOpen.x + 0.5, posOpen.y + 0.5, posOpen.z + 0.5);
			surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
		}
	}
	else{ //affected by both close and open
		//only work as open
		if (posOpen.x < 0 && posOpen.x < posOpen.z){
			float res = 0;
			surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
		}
		else{
			float res = tex3D(volumeTexInput, posOpen.x + 0.5, posOpen.y + 0.5, posOpen.z + 0.5);
			surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
		}
	}
	return;
}


__global__ void
d_updateVolumebyMatrixInfo_tunnel_rect(cudaExtent volumeSize, float3 start, float3 end, float3 spacing, float r, float deformationScaleVertical, float3 dir2nd)
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
		if (abs(l2) < deformationScaleVertical){
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

	d_updateVolumebyMatrixInfo_rect << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, degree, deformationScaleVertical, rectVerticalDir);
	checkCudaErrors(cudaUnbindTexture(volumeTexInput));
	//checkCudaErrors(cudaUnbindTexture(channelVolumeTexInput));
}


void PositionBasedDeformProcessor::doDeforme2Tunnel(float degree, float degreeClose)
{
	cudaExtent size = volume->volumeCuda.size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp(size.width, blockSize.x), iDivUp(size.height, blockSize.y), iDivUp(size.depth, blockSize.z));

	cudaChannelFormatDesc cd = volume->volumeCudaOri.channelDesc;
	checkCudaErrors(cudaBindTextureToArray(volumeTexInput, volume->volumeCudaOri.content, cd));
	checkCudaErrors(cudaBindSurfaceToArray(volumeSurfaceOut, volume->volumeCuda.content));

	d_updateVolumebyMatrixInfo_rect_2anime << <gridSize, blockSize >> >(size, volume->spacing, tunnelStart, tunnelEnd, degree, deformationScaleVertical, rectVerticalDir, lastDeformationDegree, lastDeformationDirVertical, lastTunnelStart, lastTunnelEnd, degreeClose);

	checkCudaErrors(cudaUnbindTexture(volumeTexInput));
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

	d_updateVolumebyMatrixInfo_tunnel_rect << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, deformationScale, deformationScaleVertical, rectVerticalDir);
	checkCudaErrors(cudaUnbindTexture(channelVolumeTexInput));
}


void PositionBasedDeformProcessor::computeTunnelInfo()
{
	float3 eyeInLocal = matrixMgr->getEyeInLocal();
	//float3 tunnelAxis = normalize(matrixMgr->recentMove);
	float3 tunnelAxis = normalize(matrixMgr->getViewVecInLocal());

	float3 recentMove = normalize(matrixMgr->recentMove);
	if (dot(recentMove, tunnelAxis) < -0.9){
		tunnelAxis = -tunnelAxis;
	}

	float step = 0.5;
	
	tunnelEnd = eyeInLocal + tunnelAxis*step;
	while (channelVolume->inRange(tunnelEnd) && channelVolume->getVoxel(tunnelEnd) < 0.5){
		tunnelEnd += tunnelAxis*step;
	}
	//tunnelStart = eyeInLocal;
	tunnelStart = eyeInLocal + tunnelAxis*(-2); //better also compute this number (-2)


	//rectVerticalDir = targetUpVecInLocal;
	if (abs(dot(targetUpVecInLocal, tunnelAxis)) < 0.9){
		rectVerticalDir = normalize(cross(cross(tunnelAxis, targetUpVecInLocal), tunnelAxis));
	}
	else{
		rectVerticalDir = matrixMgr->getViewVecInLocal();
	}
	//std::cout << "rectVerticalDir: " << rectVerticalDir.x << " " << rectVerticalDir.y << " " << rectVerticalDir.z << std::endl;
}


bool PositionBasedDeformProcessor::inDeformedCell(float3 pos)
{
	bool* d_inchannel;
	cudaMalloc(&d_inchannel, sizeof(bool)* 1);
	cudaChannelFormatDesc cd2 = channelVolume->volumeCudaOri.channelDesc;
	checkCudaErrors(cudaBindTextureToArray(channelVolumeTexInput, channelVolume->volumeCuda.content, cd2));
	d_posInDeformedChannelVolume << <1, 1 >> >(pos, volume->size, volume->spacing, d_inchannel);
	bool inchannel;
	cudaMemcpy(&inchannel, d_inchannel, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
	return inchannel;
}


bool PositionBasedDeformProcessor::process(float* modelview, float* projection, int winWidth, int winHeight)
{
	if (!isActive)
		return false;

	float3 eyeInLocal = matrixMgr->getEyeInLocal();

	if (lastVolumeState == ORIGINAL){
		if (volume->inRange(eyeInLocal) && channelVolume->getVoxel(eyeInLocal) < 0.5){
			// in solid area
			// in this case, set the start of deformation
			if (lastEyeState != inWall){
				//eyeFromCellToWall();
				lastVolumeState = DEFORMED;
				lastEyeState = inWall;

				computeTunnelInfo();
				doTunnelDeforme(deformationScale);
				//start a opening animation
				hasOpenAnimeStarted = true;
				hasCloseAnimeStarted = false;
				startOpen = std::clock();
			}
			else if (lastEyeState == inWall){
				//from wall to wall
			}
		}
		else{
			// either eyeInLocal is out of range, or eyeInLocal is in channel
			//in this case, no state change
		}
	}
	else{ //lastVolumeState == Deformed
		if (volume->inRange(eyeInLocal) && channelVolume->getVoxel(eyeInLocal) < 0.5){

			//in area which is solid in the original volume
			bool inchannel = inDeformedCell(eyeInLocal);
			if (inchannel){
				// not in the solid region in the deformed volume
				// in this case, no change
			}
			else{
				//even in the deformed volume, eye is still inside the solid region 
				//eye should just move to a solid region

				//volume->reset();
				//channelVolume->reset();

				lastDeformationDegree = closeStartingRadius;
				lastDeformationDirVertical = rectVerticalDir;
				lastTunnelStart = tunnelStart;
				lastTunnelEnd = tunnelEnd;

				computeTunnelInfo();
				doTunnelDeforme(deformationScale);
	
				hasOpenAnimeStarted = true;//start a opening animation
				hasCloseAnimeStarted = true; //since eye should just moved to the current solid, the previous solid should be closed 
				startOpen = std::clock();
			}
		}
		else{// in area which is channel in the original volume
			hasCloseAnimeStarted = true;
			hasOpenAnimeStarted = false;
			startClose = std::clock();

			channelVolume->reset();
			lastVolumeState = ORIGINAL;
			lastEyeState = inCell;
		}
	}

	if (hasOpenAnimeStarted && hasCloseAnimeStarted){
		float r, rClose;
		double past = (std::clock() - startOpen) / (double)CLOCKS_PER_SEC;
		if (past >= totalDuration){
			r = deformationScale;
			hasOpenAnimeStarted = false;
			hasCloseAnimeStarted = false;
			rClose = 0;
		}
		else{
			r = past / totalDuration*deformationScale;

			if (past >= closeDuration){
				hasCloseAnimeStarted = false;
				rClose = 0;
			}
			else{
				rClose = (1 - past / closeDuration)*closeStartingRadius;
			}

			doDeforme2Tunnel(r, rClose);
		}


	}
	else if (hasOpenAnimeStarted){
		float r;
		double past = (std::clock() - startOpen) / (double)CLOCKS_PER_SEC;
		if (past >= totalDuration){
			r = deformationScale;
			hasOpenAnimeStarted = false;
		}
		else{
			r = past / totalDuration*deformationScale;
			doDeforme(r);
			closeStartingRadius = r;
			closeDuration = past;
		}
	}
	else if (hasCloseAnimeStarted){
		float r;
		double past = (std::clock() - startClose) / (double)CLOCKS_PER_SEC;
		if (past >= closeDuration){
			volume->reset();
			hasCloseAnimeStarted = false;
		}
		else{
			r = (1 - past / closeDuration)*closeStartingRadius;
			doDeforme(r);
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

