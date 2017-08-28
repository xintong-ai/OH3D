#include "PositionBasedDeformProcessor2.h"
#include "Lens.h"
#include "MeshDeformProcessor.h"
#include "TransformFunc.h"
#include "MatrixManager.h"

#include "Volume.h"
#include "PolyMesh.h"
#include "Particle.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>


//!!! NOTE !!! spacing not considered yet!!!! in the global functions


texture<float, 3, cudaReadModeElementType>  volumeTexInput;
surface<void, cudaSurfaceType3D>			volumeSurfaceOut;

texture<float, 3, cudaReadModeElementType>  channelVolumeTex;
surface<void, cudaSurfaceType3D>			channelVolumeSurface;


PositionBasedDeformProcessor2::PositionBasedDeformProcessor2(std::shared_ptr<Particle> ori, std::shared_ptr<MatrixManager> _m, std::shared_ptr<Volume> ch)
{
	particle = ori;
	matrixMgr = _m;
	channelVolume = ch;
	spacing = channelVolume->spacing;

	sdkCreateTimer(&timer);
	sdkCreateTimer(&timerFrame);

	dataType = PARTICLE;

	d_vec_posOrig.assign(&(particle->pos[0]), &(particle->pos[0]) + particle->numParticles);
	d_vec_posTarget.assign(&(particle->pos[0]), &(particle->pos[0]) + particle->numParticles);
}

__device__ bool inTunnel(float3 pos, float3 start, float3 end, float deformationScale, float deformationScaleVertical, float3 dir2nd)
{
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
			if (dis < deformationScale / 2.0){
				return true;
			}
		}
	}
	return false;
}

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

			if (dis < r / 2){
				return emptyMark;
			}
			else if (dis < r){
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
d_updateVolumebyMatrixInfo_rect(cudaExtent volumeSize, float3 start, float3 end, float3 spacing, float r, float deformationScale, float deformationScaleVertical, float3 dir2nd)
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

			if (dis < r){
				float res = 0;
				surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
			}
			else if (dis < deformationScale){
				float3 prjPoint = start + l*tunnelVec + l2*dir2nd;
				float3 dir = normalize(pos - prjPoint);
				float3 samplePos = prjPoint + dir* (dis - r) / (deformationScale - r)*deformationScale;
				samplePos /= spacing;

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
	}
	else{
		float res = tex3D(volumeTexInput, x + 0.5, y + 0.5, z + 0.5);
		surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
	}
	return;
}


__global__ void
d_updateVolumebyMatrixInfo_tunnel_rect(cudaExtent volumeSize, float3 start, float3 end, float3 spacing, float r, float deformationScale, float deformationScaleVertical, float3 dir2nd)
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

			if (dis < r / 2){
				float res2 = 1;
				surf3Dwrite(res2, channelVolumeSurface, x * sizeof(float), y, z);
			}
			else if (dis < deformationScale){
				float3 prjPoint = start + l*tunnelVec + l2*dir2nd;
				float3 dir = normalize(pos - prjPoint);
				float3 samplePos = prjPoint + dir* (dis - r) / (deformationScale - r)*deformationScale;

				samplePos /= spacing;
				float res2 = tex3D(channelVolumeTex, samplePos.x, samplePos.y, samplePos.z);
				surf3Dwrite(res2, channelVolumeSurface, x * sizeof(float), y, z);
			}
			else{
				float res2 = tex3D(channelVolumeTex, x, y, z);
				surf3Dwrite(res2, channelVolumeSurface, x * sizeof(float), y, z);
			}
		}
		else{
			float res2 = tex3D(channelVolumeTex, x, y, z);
			surf3Dwrite(res2, channelVolumeSurface, x * sizeof(float), y, z);
		}
	}
	else{
		float res2 = tex3D(channelVolumeTex, x, y, z);
		surf3Dwrite(res2, channelVolumeSurface, x * sizeof(float), y, z);
	}
	return;
}

__global__ void
d_posInDeformedChannelVolume(float3 pos, int3 dims, float3 spacing, bool* inChannel)
{
	float3 ind = pos / spacing;
	if (ind.x >= 0 && ind.x < dims.x && ind.y >= 0 && ind.y < dims.y && ind.z >= 0 && ind.z<dims.z) {
		float res = tex3D(channelVolumeTex, ind.x, ind.y, ind.z);
		if (res > 0.5)
			*inChannel = true;
		else
			*inChannel = false;
	}
	else{
		*inChannel = true;
	}
}



struct functor_particleDeform
{
	int n;
	float3 start, end, dir2nd;
	float3 spacing;
	float r, deformationScale, deformationScaleVertical;

	template<typename Tuple>
	__device__ __host__ void operator() (Tuple t){//float2 screenPos, float4 clipPos) {
		float4 posf4 = thrust::get<0>(t);
		float3 pos = make_float3(posf4.x, posf4.y, posf4.z) * spacing;
		float3 newPos;
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

				if (dis < deformationScale){
					float newDis = deformationScale - (deformationScale - dis) / deformationScale * (deformationScale - r);
					newPos = prjPoint + newDis * dir;
				}
				else{
					newPos = pos;
				}
			}
			else{
				newPos = pos;
			}
		}
		else{
			newPos = pos;
		}
		thrust::get<1>(t) = make_float4(newPos.x, newPos.y, newPos.z, 1);

	}


	functor_particleDeform(int _n, float3 _start, float3 _end, float3 _spacing, float _r, float _deformationScale, float _deformationScaleVertical, float3 _dir2nd)
		: n(_n), start(_start), end(_end), spacing(_spacing), r(_r), deformationScale(_deformationScale), deformationScaleVertical(_deformationScaleVertical), dir2nd(_dir2nd){}
};

void PositionBasedDeformProcessor2::doParticleDeform(float degree)
{
	if (!deformData)
		return;
	int count = particle->numParticles;

	//for debug
	//	std::vector<float4> tt(count);
	//	//thrust::copy(tt.begin(), tt.end(), d_vec_posTarget.begin());
	//	std::cout << "pos of region 0 before: " << tt[0].x << " " << tt[0].y << " " << tt[0].z << std::endl;

	thrust::for_each(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		d_vec_posOrig.begin(),
		d_vec_posTarget.begin()
		)),
		thrust::make_zip_iterator(
		thrust::make_tuple(
		d_vec_posOrig.end(),
		d_vec_posTarget.end()
		)),
		functor_particleDeform(count, tunnelStart, tunnelEnd, channelVolume->spacing, degree, deformationScale, deformationScaleVertical, rectVerticalDir));

	thrust::copy(d_vec_posTarget.begin(), d_vec_posTarget.end(), &(particle->pos[0]));

	//	std::cout << "moved particles by: " << degree <<" with count "<<count<< std::endl;
	//	std::cout << "pos of region 0: " << particle->pos[0].x << " " << particle->pos[0].y << " " << particle->pos[0].z << std::endl;

}


void PositionBasedDeformProcessor2::doChannelVolumeDeform()
{
	cudaExtent size = channelVolume->volumeCuda.size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp(size.width, blockSize.x), iDivUp(size.height, blockSize.y), iDivUp(size.depth, blockSize.z));

	cudaChannelFormatDesc cd2 = channelVolume->volumeCuda.channelDesc;
	checkCudaErrors(cudaBindTextureToArray(channelVolumeTex, channelVolume->volumeCudaOri.content, cd2));
	checkCudaErrors(cudaBindSurfaceToArray(channelVolumeSurface, channelVolume->volumeCuda.content));

	d_updateVolumebyMatrixInfo_tunnel_rect << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, channelVolume->spacing, deformationScale, deformationScale, deformationScaleVertical, rectVerticalDir);
	checkCudaErrors(cudaUnbindTexture(channelVolumeTex));
}



void PositionBasedDeformProcessor2::computeTunnelInfo(float3 centerPoint)
{
	if (isForceDeform) //just for testing, may not be precise
	{
		//when this funciton is called, suppose we already know that centerPoint is not inWall
		float3 tunnelAxis = normalize(matrixMgr->getViewVecInLocal());
		//rectVerticalDir = targetUpVecInLocal;
		if (abs(dot(targetUpVecInLocal, tunnelAxis)) < 0.9){
			rectVerticalDir = normalize(cross(cross(tunnelAxis, targetUpVecInLocal), tunnelAxis));
		}
		else{
			rectVerticalDir = matrixMgr->getViewVecInLocal();
		}

		//old method
		float step = 0.5;
		tunnelStart = centerPoint;
		while (!channelVolume->inRange(tunnelStart / spacing) || channelVolume->getVoxel(tunnelStart / spacing) > 0.5){
			tunnelStart += tunnelAxis*step;
		}
		tunnelEnd = tunnelStart + tunnelAxis*step;
		while (channelVolume->inRange(tunnelEnd / spacing) && channelVolume->getVoxel(tunnelEnd / spacing) < 0.5){
			tunnelEnd += tunnelAxis*step;
		}



		/* //new method

		//when this funciton is called, suppose we already know that centerPoint is NOT inWall
		float3 tunnelAxis = normalize(matrixMgr->getViewVecInLocal());
		//rectVerticalDir = targetUpVecInLocal;
		if (abs(dot(targetUpVecInLocal, tunnelAxis)) < 0.9){
		rectVerticalDir = normalize(cross(cross(tunnelAxis, targetUpVecInLocal), tunnelAxis));
		}
		else{
		rectVerticalDir = matrixMgr->getViewVecInLocal();
		}

		float step = 1;

		bool* d_planeHasSolid;
		cudaMalloc(&d_planeHasSolid, sizeof(bool)* 1);
		cudaChannelFormatDesc cd2 = channelVolume->volumeCudaOri.channelDesc;
		checkCudaErrors(cudaBindTextureToArray(channelVolumeTex, channelVolume->volumeCudaOri.content, cd2));

		int ycount = ceil(deformationScale) * 2 + 1;
		int zcount = ceil(deformationScaleVertical) * 2 + 1;
		int threadsPerBlock = 64;
		int blocksPerGrid = (ycount*zcount + threadsPerBlock - 1) / threadsPerBlock;

		float3 dir_y = normalize(cross(rectVerticalDir, tunnelAxis));

		tunnelStart = centerPoint;
		bool startNotFound = true;
		while (startNotFound){
		tunnelStart += tunnelAxis*step;
		bool temp = false;
		cudaMemcpy(d_planeHasSolid, &temp, sizeof(bool)* 1, cudaMemcpyHostToDevice);
		d_checkPlane << <blocksPerGrid, threadsPerBlock >> >(tunnelStart, channelVolume->size, channelVolume->spacing, dir_y, rectVerticalDir, ycount, zcount, d_planeHasSolid);
		cudaMemcpy(&startNotFound, d_planeHasSolid, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		startNotFound = !startNotFound;
		}

		tunnelEnd = tunnelStart;
		bool endNotFound = true;
		while (endNotFound){
		tunnelEnd += tunnelAxis*step;
		bool temp = false;
		cudaMemcpy(d_planeHasSolid, &temp, sizeof(bool)* 1, cudaMemcpyHostToDevice);
		d_checkPlane << <blocksPerGrid, threadsPerBlock >> >(tunnelEnd, channelVolume->size, channelVolume->spacing, dir_y, rectVerticalDir, ycount, zcount, d_planeHasSolid);
		cudaMemcpy(&endNotFound, d_planeHasSolid, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		}

		std::cout << "tunnelStart: " << tunnelStart.x << " " << tunnelStart.y << " " << tunnelStart.z << std::endl;
		std::cout << "centerPoint: " << centerPoint.x << " " << centerPoint.y << " " << centerPoint.z << std::endl;
		std::cout << "tunnelEnd: " << tunnelEnd.x << " " << tunnelEnd.y << " " << tunnelEnd.z << std::endl << std::endl;
		cudaFree(d_planeHasSolid);
		*/
	}
	else{
		//when this funciton is called, suppose we already know that centerPoint is inWall
		float3 tunnelAxis = normalize(matrixMgr->getViewVecInLocal());
		//rectVerticalDir = targetUpVecInLocal;
		if (abs(dot(targetUpVecInLocal, tunnelAxis)) < 0.9){
			rectVerticalDir = normalize(cross(cross(tunnelAxis, targetUpVecInLocal), tunnelAxis));
		}
		else{
			rectVerticalDir = matrixMgr->getViewVecInLocal();
		}


		//old method
		float step = 0.5;
		tunnelEnd = centerPoint + tunnelAxis*step;
		while (channelVolume->inRange(tunnelEnd / spacing) && channelVolume->getVoxel(tunnelEnd / spacing) < 0.5){
			tunnelEnd += tunnelAxis*step;
		}
		tunnelStart = centerPoint;
		while (channelVolume->inRange(tunnelStart / spacing) && channelVolume->getVoxel(tunnelStart / spacing) < 0.5){
			tunnelStart -= tunnelAxis*step;
		}


		/* //new method
		float step = 1;

		bool* d_planeHasSolid;
		cudaMalloc(&d_planeHasSolid, sizeof(bool)* 1);
		cudaChannelFormatDesc cd2 = channelVolume->volumeCudaOri.channelDesc;
		checkCudaErrors(cudaBindTextureToArray(channelVolumeTex, channelVolume->volumeCudaOri.content, cd2));

		int ycount = ceil(deformationScale) * 2 + 1;
		int zcount = ceil(deformationScaleVertical) * 2 + 1;
		int threadsPerBlock = 64;
		int blocksPerGrid = (ycount*zcount + threadsPerBlock - 1) / threadsPerBlock;

		float3 dir_y = normalize(cross(rectVerticalDir, tunnelAxis));

		tunnelStart = centerPoint;
		bool startNotFound = true;
		while (startNotFound){
		tunnelStart -= tunnelAxis*step;
		bool temp = false;
		cudaMemcpy(d_planeHasSolid, &temp, sizeof(bool)* 1, cudaMemcpyHostToDevice);
		d_checkPlane << <blocksPerGrid, threadsPerBlock >> >(tunnelStart, channelVolume->size, channelVolume->spacing, dir_y, rectVerticalDir, ycount, zcount, d_planeHasSolid);
		cudaMemcpy(&startNotFound, d_planeHasSolid, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		}

		tunnelEnd = centerPoint;
		bool endNotFound = true;
		while (endNotFound){
		tunnelEnd += tunnelAxis*step;
		bool temp = false;
		cudaMemcpy(d_planeHasSolid, &temp, sizeof(bool)* 1, cudaMemcpyHostToDevice);
		d_checkPlane << <blocksPerGrid, threadsPerBlock >> >(tunnelEnd, channelVolume->size, channelVolume->spacing, dir_y, rectVerticalDir, ycount, zcount, d_planeHasSolid);
		cudaMemcpy(&endNotFound, d_planeHasSolid, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		}

		//std::cout << "tunnelStart: " << tunnelStart.x << " " << tunnelStart.y << " " << tunnelStart.z << std::endl;
		//std::cout << "centerPoint: " << centerPoint.x << " " << centerPoint.y << " " << centerPoint.z << std::endl;
		//std::cout << "tunnelEnd: " << tunnelEnd.x << " " << tunnelEnd.y << " " << tunnelEnd.z << std::endl << std::endl;
		cudaFree(d_planeHasSolid);
		*/
	}
}


bool PositionBasedDeformProcessor2::inDeformedCell(float3 pos)
{
	bool* d_inchannel;
	cudaMalloc(&d_inchannel, sizeof(bool)* 1);
	cudaChannelFormatDesc cd2 = channelVolume->volumeCudaOri.channelDesc;
	checkCudaErrors(cudaBindTextureToArray(channelVolumeTex, channelVolume->volumeCuda.content, cd2));
	d_posInDeformedChannelVolume << <1, 1 >> >(pos, channelVolume->size, channelVolume->spacing, d_inchannel);
	bool inchannel;
	cudaMemcpy(&inchannel, d_inchannel, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
	cudaFree(d_inchannel);
	return inchannel;
}


bool PositionBasedDeformProcessor2::inRange(float3 v)
{
	if (dataType == VOLUME){
		return false;
	}
	else if (dataType == MESH){
		return false;
	}
	else if (dataType == PARTICLE){
		return channelVolume->inRange(v / spacing); //actually currently channelVolume->inRange will serve all possibilities. Keep 3 cases in case of unexpected needs.
	}
	else{
		std::cout << " inRange not implemented " << std::endl;
		exit(0);
	}
}

void PositionBasedDeformProcessor2::deformDataByDegree(float r)
{
	if (dataType == VOLUME){
	}
	else if (dataType == MESH){
	}
	else if (dataType == PARTICLE){
		doParticleDeform(r);
	}
	else{
		std::cout << " inRange not implemented " << std::endl;
		exit(0);
	}
}

void PositionBasedDeformProcessor2::deformDataByDegree2Tunnel(float r, float rClose)
{
	return;
}

void PositionBasedDeformProcessor2::resetData()
{
	if (dataType == VOLUME){
	}
	else if (dataType == MESH){
	}
	else if (dataType == PARTICLE){
		particle->reset();
		d_vec_posOrig.assign(&(particle->pos[0]), &(particle->pos[0]) + particle->numParticles);
		d_vec_posTarget.assign(&(particle->pos[0]), &(particle->pos[0]) + particle->numParticles);
	}
	else{
		std::cout << " inRange not implemented " << std::endl;
		exit(0);
	}
}


bool PositionBasedDeformProcessor2::process(float* modelview, float* projection, int winWidth, int winHeight)
{
	if (!isActive)
		return false;

	float3 eyeInLocal = matrixMgr->getEyeInLocal();

	if (lastDataState == ORIGINAL){
		if (isForceDeform){
			//if (lastEyeState != inWall){
			lastDataState = DEFORMED;
			//lastEyeState = inWall;

			computeTunnelInfo(eyeInLocal);
			doChannelVolumeDeform();


			//start a opening animation
			hasOpenAnimeStarted = true;
			hasCloseAnimeStarted = false; //currently if there is closing procedure for other tunnels, they are finished suddenly
			startOpen = std::clock();
			//}
			//else if (lastEyeState == inWall){
			//from wall to wall
			//}
		}
		else if (inRange(eyeInLocal) && channelVolume->getVoxel(eyeInLocal / spacing) < 0.5){
			// in solid area
			// in this case, set the start of deformation
			if (lastEyeState != inWall){
				lastDataState = DEFORMED;
				lastEyeState = inWall;

				computeTunnelInfo(eyeInLocal);
				doChannelVolumeDeform();

				//start a opening animation
				hasOpenAnimeStarted = true;
				hasCloseAnimeStarted = false; //currently if there is closing procedure for other tunnels, they are finished suddenly
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
	else{ //lastDataState == Deformed
		if (isForceDeform){

		}
		else if (inRange(eyeInLocal) && channelVolume->getVoxel(eyeInLocal / spacing) < 0.5){
			//in area which is solid in the original volume
			bool inchannel = inDeformedCell(eyeInLocal);
			if (inchannel){
				// not in the solid region in the deformed volume
				// in this case, no change
			}
			else{
				//std::cout <<"Triggered "<< lastDataState << " " << lastEyeState << " " << hasOpenAnimeStarted << " " << hasCloseAnimeStarted << std::endl;
				//even in the deformed volume, eye is still inside the solid region 
				//eye should just move to a solid region

				//poly->reset();
				//channelVolume->reset();

				sdkResetTimer(&timer);
				sdkStartTimer(&timer);

				sdkResetTimer(&timerFrame);

				fpsCount = 0;

				lastOpenFinalDegree = closeStartingRadius;
				lastDeformationDirVertical = rectVerticalDir;
				lastTunnelStart = tunnelStart;
				lastTunnelEnd = tunnelEnd;

				computeTunnelInfo(eyeInLocal);
				doChannelVolumeDeform();

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
			lastDataState = ORIGINAL;
			lastEyeState = inCell;
		}
	}

	if (hasOpenAnimeStarted && hasCloseAnimeStarted){
		//std::cout << "processing as wanted" << std::endl;
		float rClose;
		double past = (std::clock() - startOpen) / (double)CLOCKS_PER_SEC;
		if (past >= totalDuration){
			r = deformationScale / 2;
			hasOpenAnimeStarted = false;
			hasCloseAnimeStarted = false;

			sdkStopTimer(&timer);
			std::cout << "Mixed animation fps: " << fpsCount / (sdkGetAverageTimerValue(&timer) / 1000.f) << std::endl;

			sdkStopTimer(&timer);
			std::cout << "Mixed animation cost each frame: " << sdkGetAverageTimerValue(&timerFrame) << " ms" << std::endl;
		}
		else{
			sdkStartTimer(&timerFrame);
			fpsCount++;

			r = past / totalDuration*deformationScale / 2;
			if (past >= closeDuration){
				hasCloseAnimeStarted = false;
				rClose = 0;
				deformDataByDegree(r);
			}
			else{
				rClose = (1 - past / closeDuration)*closeStartingRadius;
				//doVolumeDeform2Tunnel(r, rClose);  //TO BE IMPLEMENTED
				deformDataByDegree2Tunnel(r, rClose);
			}

			sdkStopTimer(&timerFrame);
		}
	}
	else if (hasOpenAnimeStarted){
		//std::cout << "doing openning" << std::endl;

		double past = (std::clock() - startOpen) / (double)CLOCKS_PER_SEC;
		if (past >= totalDuration){
			r = deformationScale / 2;
			hasOpenAnimeStarted = false;
			//closeStartingRadius = r;
			closeDuration = totalDuration;//or else closeDuration may be less than totalDuration
		}
		else{
			r = past / totalDuration*deformationScale / 2;
			deformDataByDegree(r);
			closeStartingRadius = r;
			closeDuration = past;
		}
	}
	else if (hasCloseAnimeStarted){
		//std::cout << "doing closing" << std::endl;

		double past = (std::clock() - startClose) / (double)CLOCKS_PER_SEC;
		if (past >= closeDuration){
			resetData();
			hasCloseAnimeStarted = false;
			r = 0;
		}
		else{
			r = (1 - past / closeDuration)*closeStartingRadius;
			deformDataByDegree(r);
		}
	}

	return false;
}



void PositionBasedDeformProcessor2::InitCudaSupplies()
{
	volumeTexInput.normalized = false;
	volumeTexInput.filterMode = cudaFilterModeLinear;
	volumeTexInput.addressMode[0] = cudaAddressModeBorder;
	volumeTexInput.addressMode[1] = cudaAddressModeBorder;
	volumeTexInput.addressMode[2] = cudaAddressModeBorder;

	channelVolumeTex.normalized = false;
	channelVolumeTex.filterMode = cudaFilterModePoint;
	channelVolumeTex.addressMode[0] = cudaAddressModeBorder;
	channelVolumeTex.addressMode[1] = cudaAddressModeBorder;
	channelVolumeTex.addressMode[2] = cudaAddressModeBorder;

	volumeCudaIntermediate = std::make_shared<VolumeCUDA>();
	volumeCudaIntermediate->VolumeCUDA_deinit();
	volumeCudaIntermediate->VolumeCUDA_init(channelVolume->size, channelVolume->values, 1, 1);
	//	volumeCudaIntermediate.VolumeCUDA_init(volume->size, 0, 1, 1);//??
}

