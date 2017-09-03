#include "PositionBasedDeformProcessor.h"
#include "TransformFunc.h"
#include "MatrixManager.h"

#include "Volume.h"
#include "PolyMesh.h"
#include "Particle.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>


#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

//!!! NOTE !!! spacing not considered yet!!!! in the global functions

// 1D transfer function texture
//same to the transferTex in VolumeRendererCUDAKernel, but renamed
texture<float4, 1, cudaReadModeElementType>           transferTex2;

texture<float, 3, cudaReadModeElementType>  volumeTexInput;
texture<float, 3, cudaReadModeElementType>  volumePointTexture;
surface<void, cudaSurfaceType3D>			volumeSurfaceOut;

#include "myDefineRayCasting.h"


PositionBasedDeformProcessor::PositionBasedDeformProcessor(std::shared_ptr<Volume> ori, std::shared_ptr<MatrixManager> _m)
{
	volume = ori;
	matrixMgr = _m;
	InitCudaSupplies();
	sdkCreateTimer(&timer);
	sdkCreateTimer(&timerFrame);

	minPos = make_float3(0, 0, 0);
	maxPos = make_float3(volume->size.x, volume->size.y, volume->size.z);

	dataType = VOLUME;
};

PositionBasedDeformProcessor::PositionBasedDeformProcessor(std::shared_ptr<PolyMesh> ori, std::shared_ptr<MatrixManager> _m)
{
	poly = ori;
	matrixMgr = _m;

	sdkCreateTimer(&timer);
	sdkCreateTimer(&timerFrame);

	dataType = MESH;

	//NOTE!! here doubled the space. Hopefully it is large enough
	cudaMalloc(&d_vertexCoords, sizeof(float)*poly->vertexcount * 3 * 2);
	cudaMalloc(&d_norms, sizeof(float)*poly->vertexcount * 3 * 2);
	cudaMalloc(&d_vertexCoords_init, sizeof(float)*poly->vertexcount * 3 * 2);
	cudaMalloc(&d_indices, sizeof(unsigned int)*poly->facecount * 3 * 2);
	cudaMalloc(&d_numAddedFaces, sizeof(int));
	cudaMalloc(&d_vertexDeviateVals, sizeof(float)*poly->vertexcount * 2);
	cudaMalloc(&d_vertexColorVals, sizeof(float)*poly->vertexcount * 2);

	cudaMemcpy(d_vertexCoords_init, poly->vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, poly->indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyHostToDevice);
};

PositionBasedDeformProcessor::PositionBasedDeformProcessor(std::shared_ptr<Particle> ori, std::shared_ptr<MatrixManager> _m)
{
	particle = ori;
	matrixMgr = _m;

	InitCudaSupplies();
	sdkCreateTimer(&timer);
	sdkCreateTimer(&timerFrame);

	dataType = PARTICLE;

	d_vec_posOrig.assign(&(particle->pos[0]), &(particle->pos[0]) + particle->numParticles);
	d_vec_posTarget.assign(&(particle->pos[0]), &(particle->pos[0]) + particle->numParticles);
}

void PositionBasedDeformProcessor::updateParticleData(std::shared_ptr<Particle> ori)
{
	particle = ori;
	d_vec_posOrig.assign(&(particle->pos[0]), &(particle->pos[0]) + particle->numParticles);

	//process(0, 0, 0, 0);

	if (lastDataState == DEFORMED && isActive){
		doParticleDeform(r);
	}
	else{
		d_vec_posTarget.assign(&(particle->pos[0]), &(particle->pos[0]) + particle->numParticles);
	}
}


__device__ bool d_inTunnel(float3 pos, float3 start, float3 end, float deformationScale, float deformationScaleVertical, float3 dir2nd)
{
	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);
	float3 voxelVec = pos - start;
	float l = dot(voxelVec, tunnelVec);
	if (l > 0 && l < tunnelLength){
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


__global__ void d_updatePolyMeshbyMatrixInfo_rect(float* vertexCoords_init, float* vertexCoords, int vertexcount,
	float3 start, float3 end , float r, float deformationScale, float deformationScaleVertical, float3 dir2nd,
	float* vertexDeviateVals)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= vertexcount)	return;
	vertexDeviateVals[i] = 0;

	float3 pos = make_float3(vertexCoords_init[3 * i], vertexCoords_init[3 * i + 1], vertexCoords_init[3 * i + 2]);
	vertexCoords[3 * i] = pos.x;
	vertexCoords[3 * i + 1] = pos.y;
	vertexCoords[3 * i + 2] = pos.z;

	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);

	float3 voxelVec = pos - start;
	float l = dot(voxelVec, tunnelVec);
	if (l > 0 && l < tunnelLength){
		float l2 = dot(voxelVec, dir2nd);
		if (abs(l2) < deformationScaleVertical){
			float3 prjPoint = start + l*tunnelVec + l2*dir2nd;
			float dis = length(pos - prjPoint);

			//!!NOTE!! the case dis==0 is not processed!! suppose this case will not happen by some spacial preprocessing
			if (dis > 0 && dis < deformationScale){
				float3 dir = normalize(pos - prjPoint);

				float newDis = deformationScale - (deformationScale - dis) / deformationScale * (deformationScale - r);
				float3 newPos = prjPoint + newDis * dir;
				vertexCoords[3 * i] = newPos.x;
				vertexCoords[3 * i + 1] = newPos.y;
				vertexCoords[3 * i + 2] = newPos.z;

				vertexDeviateVals[i] = length(newPos - pos) / (deformationScale / 2); //value range [0,1]
			}
		}
	}

	return;
}

void PositionBasedDeformProcessor::doPolyDeform(float degree)
{
	if (!deformData)
		return;
	int threadsPerBlock = 64;
	int blocksPerGrid = (poly->vertexcount + threadsPerBlock - 1) / threadsPerBlock;

	d_updatePolyMeshbyMatrixInfo_rect << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords_init, d_vertexCoords, poly->vertexcount,
		tunnelStart, tunnelEnd, degree, deformationScale, deformationScaleVertical, rectVerticalDir, d_vertexDeviateVals);

	cudaMemcpy(poly->vertexCoords, d_vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);
	if (isColoringDeformedPart)
	{
		cudaMemcpy(poly->vertexDeviateVals, d_vertexDeviateVals, sizeof(float)*poly->vertexcount, cudaMemcpyDeviceToHost);
	}
}

struct functor_particleDeform
{
	int n;
	float3 start, end, dir2nd;
	float r, deformationScale, deformationScaleVertical;

	template<typename Tuple>
	__device__ __host__ void operator() (Tuple t){
		float4 posf4 = thrust::get<0>(t);
		float3 pos = make_float3(posf4.x, posf4.y, posf4.z);
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


	functor_particleDeform(int _n, float3 _start, float3 _end, float _r, float _deformationScale, float _deformationScaleVertical, float3 _dir2nd)
		: n(_n), start(_start), end(_end), r(_r), deformationScale(_deformationScale), deformationScaleVertical(_deformationScaleVertical), dir2nd(_dir2nd){}
};

void PositionBasedDeformProcessor::doParticleDeform(float degree)
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
		functor_particleDeform(count, tunnelStart, tunnelEnd, degree, deformationScale, deformationScaleVertical, rectVerticalDir));

	thrust::copy(d_vec_posTarget.begin(), d_vec_posTarget.end(), &(particle->pos[0]));

	//	std::cout << "moved particles by: " << degree <<" with count "<<count<< std::endl;
	//	std::cout << "pos of region 0: " << particle->pos[0].x << " " << particle->pos[0].y << " " << particle->pos[0].z << std::endl;

}


void PositionBasedDeformProcessor::doVolumeDeform(float degree)
{
	if (!deformData)
		return;

	cudaExtent size = volume->volumeCuda.size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp(size.width, blockSize.x), iDivUp(size.height, blockSize.y), iDivUp(size.depth, blockSize.z));

	cudaChannelFormatDesc cd = volume->volumeCudaOri.channelDesc;
	checkCudaErrors(cudaBindTextureToArray(volumeTexInput, volume->volumeCudaOri.content, cd));
	checkCudaErrors(cudaBindSurfaceToArray(volumeSurfaceOut, volume->volumeCuda.content));

	d_updateVolumebyMatrixInfo_rect << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, degree, deformationScale, deformationScaleVertical, rectVerticalDir);
	checkCudaErrors(cudaUnbindTexture(volumeTexInput));
}

void PositionBasedDeformProcessor::doVolumeDeform2Tunnel(float degree, float degreeClose)
{
	cudaExtent size = volume->volumeCuda.size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp(size.width, blockSize.x), iDivUp(size.height, blockSize.y), iDivUp(size.depth, blockSize.z));

	cudaChannelFormatDesc cd = volume->volumeCudaOri.channelDesc;

	checkCudaErrors(cudaBindTextureToArray(volumeTexInput, volume->volumeCudaOri.content, cd));
	checkCudaErrors(cudaBindSurfaceToArray(volumeSurfaceOut, volumeCudaIntermediate->content));
	d_updateVolumebyMatrixInfo_rect << <gridSize, blockSize >> >(size, lastTunnelStart, lastTunnelEnd, volume->spacing, degreeClose, deformationScale, deformationScaleVertical, lastDeformationDirVertical);
	checkCudaErrors(cudaUnbindTexture(volumeTexInput));

	checkCudaErrors(cudaBindTextureToArray(volumeTexInput, volumeCudaIntermediate->content, cd));
	checkCudaErrors(cudaBindSurfaceToArray(volumeSurfaceOut, volume->volumeCuda.content));
	d_updateVolumebyMatrixInfo_rect << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, degree, deformationScale, deformationScaleVertical, rectVerticalDir); //this function is not changed for time varying particle data yet

	checkCudaErrors(cudaUnbindTexture(volumeTexInput));

}



__global__ void
d_checkPlane(float3 planeCenter, int3 size, float3 dir_y, float3 dir_z, int ycount, int zcount, bool* d_inchannel)
{
	//int i = blockDim.x * blockIdx.x + threadIdx.x;
	//if (i >= ycount * zcount)	return;

	//int z = i / ycount;
	//int y = i - z * ycount;
	//z = z - zcount / 2;
	//y = y - ycount / 2;
	//float3 v = planeCenter + y*dir_y + z*dir_z;

	////assume spacing (1,1,1)
	//if (v.x >= 0 && v.x < size.x && v.y >= 0 && v.y < size.y && v.z >= 0 && v.z < size.z){
	//	float res = tex3D(volumeTex, v.x, v.y, v.z);
	//	if (res < 0.5){
	//		*d_inchannel = true;
	//	}
	//}
}

void PositionBasedDeformProcessor::computeTunnelInfo(float3 centerPoint)
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
		float step = 1;
		tunnelStart = centerPoint;
		while (!inRange(tunnelStart) || atProperLocationInOriData(tunnelStart)){
			tunnelStart += tunnelAxis*step;
		}
		tunnelEnd = tunnelStart + tunnelAxis*step;
		while (inRange(tunnelEnd) && !atProperLocationInOriData(tunnelEnd)){
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
		cudaChannelFormatDesc cd2 = volume->volumeCudaOri.channelDesc;
		checkCudaErrors(cudaBindTextureToArray(channelVolumeTex, volume->volumeCudaOri.content, cd2));

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
		d_checkPlane << <blocksPerGrid, threadsPerBlock >> >(tunnelStart, volume->size,  dir_y, rectVerticalDir, ycount, zcount, d_planeHasSolid);
		cudaMemcpy(&startNotFound, d_planeHasSolid, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		startNotFound = !startNotFound;
		}

		tunnelEnd = tunnelStart;
		bool endNotFound = true;
		while (endNotFound){
		tunnelEnd += tunnelAxis*step;
		bool temp = false;
		cudaMemcpy(d_planeHasSolid, &temp, sizeof(bool)* 1, cudaMemcpyHostToDevice);
		d_checkPlane << <blocksPerGrid, threadsPerBlock >> >(tunnelEnd, volume->size, dir_y, rectVerticalDir, ycount, zcount, d_planeHasSolid);
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
		float step = 1;
		tunnelEnd = centerPoint + tunnelAxis*step;
		while (inRange(tunnelEnd) && !atProperLocationInOriData(tunnelEnd )){
			tunnelEnd += tunnelAxis*step;
		}
		tunnelStart = centerPoint;
		while (inRange(tunnelStart) && !atProperLocationInOriData(tunnelStart)){
			tunnelStart -= tunnelAxis*step;
		}


		/* //new method
		float step = 1;

		bool* d_planeHasSolid;
		cudaMalloc(&d_planeHasSolid, sizeof(bool)* 1);
		cudaChannelFormatDesc cd2 = volume->volumeCudaOri.channelDesc;
		checkCudaErrors(cudaBindTextureToArray(channelVolumeTex, volume->volumeCudaOri.content, cd2));

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
		d_checkPlane << <blocksPerGrid, threadsPerBlock >> >(tunnelStart, volume->size, dir_y, rectVerticalDir, ycount, zcount, d_planeHasSolid);
		cudaMemcpy(&startNotFound, d_planeHasSolid, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		}

		tunnelEnd = centerPoint;
		bool endNotFound = true;
		while (endNotFound){
		tunnelEnd += tunnelAxis*step;
		bool temp = false;
		cudaMemcpy(d_planeHasSolid, &temp, sizeof(bool)* 1, cudaMemcpyHostToDevice);
		d_checkPlane << <blocksPerGrid, threadsPerBlock >> >(tunnelEnd, volume->size, dir_y, rectVerticalDir, ycount, zcount, d_planeHasSolid);
		cudaMemcpy(&endNotFound, d_planeHasSolid, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		}

		//std::cout << "tunnelStart: " << tunnelStart.x << " " << tunnelStart.y << " " << tunnelStart.z << std::endl;
		//std::cout << "centerPoint: " << centerPoint.x << " " << centerPoint.y << " " << centerPoint.z << std::endl;
		//std::cout << "tunnelEnd: " << tunnelEnd.x << " " << tunnelEnd.y << " " << tunnelEnd.z << std::endl << std::endl;
		cudaFree(d_planeHasSolid);
		*/
	}
}


__global__ void
d_posInSafePositionOfVolume(float3 pos, int3 dims, float3 spacing, bool* atProper, float densityThr, int checkRadius)
{
	float3 ind = pos / spacing;
	if (checkRadius == 1){
		if (ind.x >= 0 && ind.x < dims.x && ind.y >= 0 && ind.y < dims.y && ind.z >= 0 && ind.z < dims.z) {
			float4 col = tex1D(transferTex2, tex3D(volumePointTexture, ind.x, ind.y, ind.z));

			if (col.w <= densityThr)
				*atProper = true;
			else
				*atProper = false;
		}
		else{
			*atProper = true;
		}
	}
	else{
		float r = checkRadius - 1;
		int xstart = max(0.0, ind.x - r), xend = min(dims.x - 1.0, ind.x + r);
		int ystart = max(0.0, ind.y - r), yend = min(dims.y - 1.0, ind.y + r);
		int zstart = max(0.0, ind.z - r), zend = min(dims.z - 1.0, ind.z + r);
		for (int i = xstart; i <= xend; i++){
			for (int j = ystart; j <= yend; j++){
				for (int k = zstart; k <= zend; k++){
					float4 col = tex1D(transferTex2, tex3D(volumePointTexture, ind.x, ind.y, ind.z));
					if (col.w > densityThr){
						*atProper = false;
						return;
					}
				}
			}
		}
		*atProper = true;
	}
}



struct functor_dis
{
	float3 pos;
	__device__ __host__ float operator() (float4 posf4){//float2 screenPos, float4 clipPos) {
		float dis = length(make_float3(posf4.x, posf4.y, posf4.z) - pos);
		return dis;
	}
	functor_dis(float3 _pos)
		: pos(_pos){}
};


__device__ float d_disToTri(float3 p, float3 p1, float3 p2, float3 p3, float thr)
{
	float3 e12 = p2 - p1;
	float3 e23 = p3 - p2;
	float3 e31 = p1 - p3;

	float3 n = normalize(cross(e12, -e31));
	float disToPlane = dot(p - p1, n);
	if (abs(disToPlane) >= thr){
		return thr + 1; //no need further computation
	}
	float3 proj = p - n*disToPlane;

	bool isInside = false;
	if (dot(cross(e12, -e31), cross(e12, proj - p1)) >= 0){
		if (dot(cross(e23, -e12), cross(e23, proj - p2)) >= 0){
			if (dot(cross(e31, -e23), cross(e31, proj - p3)) >= 0){
				isInside = true;
			}
		}
	}
	if (isInside){
		return abs(disToPlane);
	}
	float disInPlaneSqu = min(min(dot(proj - p1, proj - p1), dot(proj - p2, proj - p2)), dot(proj - p3, proj - p3));
	float d = dot(proj - p1, e12);
	if (d > 0 && d < dot(e12, e12)){
		float projL = d / length(e12);
		disInPlaneSqu = min(disInPlaneSqu, dot(proj - p1, proj - p1) - projL*projL);
	}
	d = dot(proj - p2, e23);
	if (d > 0 && d < dot(e23, e23)){
		float projL = d / length(e23);
		disInPlaneSqu = min(disInPlaneSqu, dot(proj - p2, proj - p2) - projL*projL);
	}
	d = dot(proj - p3, e31);
	if (d > 0 && d < dot(e31, e31)){
		float projL = d / length(e31);
		disInPlaneSqu = min(disInPlaneSqu, dot(proj - p3, proj - p3) - projL*projL);
	}
	return sqrt(disInPlaneSqu + disToPlane*disToPlane);
}

__global__ void d_checkIfTooCloseToPoly(float3 pos, uint* indices, int faceCoords, float* vertexCoords, float thr, bool* res)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= faceCoords)	return;

	uint3 inds = make_uint3(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);
	float3 v1 = make_float3(vertexCoords[3 * inds.x], vertexCoords[3 * inds.x + 1], vertexCoords[3 * inds.x + 2]);
	float3 v2 = make_float3(vertexCoords[3 * inds.y], vertexCoords[3 * inds.y + 1], vertexCoords[3 * inds.y + 2]);
	float3 v3 = make_float3(vertexCoords[3 * inds.z], vertexCoords[3 * inds.z + 1], vertexCoords[3 * inds.z + 2]);

	float dis = min(min(length(pos - v1), length(pos - v2)), length(pos - v3));
	//if (d_disToTri(pos, v1, v2, v3, thr) < thr)
	if (dis < thr)
	{
		*res = true;
	}

	return;
}

bool PositionBasedDeformProcessor::atProperLocationInOriData(float3 pos)
{
	if (dataType == VOLUME){

		cudaChannelFormatDesc channelFloat4 = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaBindTextureToArray(transferTex2, rcp->d_transferFunc, channelFloat4));

		bool* d_atProper;
		cudaMalloc(&d_atProper, sizeof(bool)* 1);
		cudaChannelFormatDesc cd2 = volume->volumeCudaOri.channelDesc;
		checkCudaErrors(cudaBindTextureToArray(volumePointTexture, volume->volumeCudaOri.content, cd2));
		d_posInSafePositionOfVolume << <1, 1 >> >(pos, volume->size, volume->spacing, d_atProper, densityThr, checkRadius);
		bool atProper;
		cudaMemcpy(&atProper, d_atProper, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		cudaFree(d_atProper);
		return atProper;
	}
	else if (dataType == MESH){

		bool* d_tooCloseToData;
		cudaMalloc(&d_tooCloseToData, sizeof(bool)* 1);
		cudaMemset(d_tooCloseToData, 0, sizeof(bool)* 1);

		int threadsPerBlock = 64;
		int blocksPerGrid = (poly->facecount + threadsPerBlock - 1) / threadsPerBlock;

		d_checkIfTooCloseToPoly << <blocksPerGrid, threadsPerBlock >> >(pos, d_indices, poly->facecount, d_vertexCoords_init, disThr, d_tooCloseToData);

		bool tooCloseToData;
		cudaMemcpy(&tooCloseToData, d_tooCloseToData, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		cudaFree(d_tooCloseToData);
		return !tooCloseToData;
	}
	else if (dataType == PARTICLE){
		float init = 10000;
		float inSavePosition =
			thrust::transform_reduce(
			d_vec_posOrig.begin(),
			d_vec_posOrig.end(),
			functor_dis(pos),
			init,
			thrust::minimum<float>());
		return (inSavePosition > disThr);
	}
	else{
		std::cout << " in data not implemented " << std::endl;
		exit(0);
	}


}


bool PositionBasedDeformProcessor::atProperLocationInDeformedData(float3 pos)
{

	if (lastDataState == DEFORMED){ 
		//first check if inside the deform frame	
		float3 tunnelVec = normalize(tunnelEnd - tunnelStart);
		float tunnelLength = length(tunnelEnd - tunnelStart);
		float3 n = normalize(cross(rectVerticalDir, tunnelVec));
		float3 voxelVec = pos - tunnelStart;
		float l = dot(voxelVec, tunnelVec);
		if (l >= 0 && l <= tunnelLength){
			float l2 = dot(voxelVec, rectVerticalDir);
			if (abs(l2) < deformationScaleVertical){
				float l3 = dot(voxelVec, n);
				if (abs(l3) < deformationScale / 2){
					return true;
				}
			}
		}
	}
	else{
		std::cout << "should not use atProperLocationInDeformedData function !!!" << std::endl;
		exit(0);
	}

	if (dataType == VOLUME){
		cudaChannelFormatDesc channelFloat4 = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaBindTextureToArray(transferTex2, rcp->d_transferFunc, channelFloat4));

		bool* d_atProper;
		cudaMalloc(&d_atProper, sizeof(bool)* 1);
		cudaChannelFormatDesc cd2 = volume->volumeCudaOri.channelDesc;
		checkCudaErrors(cudaBindTextureToArray(volumePointTexture, volume->volumeCuda.content, cd2));

		d_posInSafePositionOfVolume << <1, 1 >> >(pos, volume->size, volume->spacing, d_atProper, densityThr, checkRadius);
		bool atProper;
		cudaMemcpy(&atProper, d_atProper, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		cudaFree(d_atProper);
		return atProper;
	}
	else if (dataType == MESH){

		bool* d_tooCloseToData;
		cudaMalloc(&d_tooCloseToData, sizeof(bool)* 1);
		cudaMemset(d_tooCloseToData, 0, sizeof(bool)* 1);

		int threadsPerBlock = 64;
		int blocksPerGrid = (poly->facecount + threadsPerBlock - 1) / threadsPerBlock;


		d_checkIfTooCloseToPoly << <blocksPerGrid, threadsPerBlock >> >(pos, d_indices, poly->facecount, d_vertexCoords, disThr, d_tooCloseToData);

		bool tooCloseToData;
		cudaMemcpy(&tooCloseToData, d_tooCloseToData, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		cudaFree(d_tooCloseToData);
		return !tooCloseToData;

	}
	else if (dataType == PARTICLE){
		float init = 10000;
		float inSavePosition =
			thrust::transform_reduce(
			d_vec_posTarget.begin(),
			d_vec_posTarget.end(),
			functor_dis(pos),
			init,
			thrust::minimum<float>());
		return (inSavePosition > disThr);
	}
	else{
		std::cout << " in data not implemented " << std::endl;
		exit(0);
	}

}


__global__ void d_disturbVertex(float* vertexCoords, int vertexcount,
	float3 start, float3 end, float deformationScaleVertical, float3 dir2nd)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= vertexcount)	return;

	float3 pos = make_float3(vertexCoords[3 * i], vertexCoords[3 * i + 1], vertexCoords[3 * i + 2]);
	vertexCoords[3 * i] = pos.x;
	vertexCoords[3 * i + 1] = pos.y;
	vertexCoords[3 * i + 2] = pos.z;

	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);

	float thr = 0.000001;
	float disturb = 0.00001;

	float3 n = normalize(cross(dir2nd, tunnelVec));
	float3 disturbVec = n*disturb;

	float3 voxelVec = pos - start;
	float l = dot(voxelVec, tunnelVec);
	if (l > 0 && l < tunnelLength){
		float l2 = dot(voxelVec, dir2nd);
		if (abs(l2) < deformationScaleVertical){
			float3 prjPoint = start + l*tunnelVec + l2*dir2nd;
			float dis = length(pos - prjPoint);

			//when dis==0 , disturb the vertex a little to avoid numerical error
			if (dis < thr){
				vertexCoords[3 * i] += disturbVec.x;
				vertexCoords[3 * i + 1] += disturbVec.y;
				vertexCoords[3 * i + 2] += disturbVec.z;
			}
		}
	}

	return;
}

__global__ void d_modifyMesh(float* vertexCoords, unsigned int* indices, int facecount, int vertexcount, float* norms, float3 start, float3 end, float r, float deformationScale, float deformationScaleVertical, float3 dir2nd, int* numAddedFaces, float* vertexColorVals)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= facecount)	return;


	uint3 inds = make_uint3(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);
	float3 v1 = make_float3(vertexCoords[3 * inds.x], vertexCoords[3 * inds.x + 1], vertexCoords[3 * inds.x + 2]);
	float3 v2 = make_float3(vertexCoords[3 * inds.y], vertexCoords[3 * inds.y + 1], vertexCoords[3 * inds.y + 2]);
	float3 v3 = make_float3(vertexCoords[3 * inds.z], vertexCoords[3 * inds.z + 1], vertexCoords[3 * inds.z + 2]);

	float3 norm1 = make_float3(norms[3 * inds.x], norms[3 * inds.x + 1], norms[3 * inds.x + 2]);
	float3 norm2 = make_float3(norms[3 * inds.y], norms[3 * inds.y + 1], norms[3 * inds.y + 2]);
	float3 norm3 = make_float3(norms[3 * inds.z], norms[3 * inds.z + 1], norms[3 * inds.z + 2]);

	//suppose any 2 points of the triangle are not overlapping
	float dis12 = length(v2 - v1);
	float3 l12 = normalize(v2 - v1);
	float dis23 = length(v3 - v2);
	float3 l23 = normalize(v3 - v2);
	float dis31 = length(v1 - v3);
	float3 l31 = normalize(v1 - v3);

	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);

	//https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
	float3 n = normalize(cross(dir2nd, tunnelVec));
	bool para12 = abs(dot(l12, n)) < 0.000001;
	bool para23 = abs(dot(l23, n)) < 0.000001;
	bool para31 = abs(dot(l31, n)) < 0.000001;
	float d12intersect = dot(start - v1, n) / (para12 ? 0.000001 : dot(l12, n));
	float d23intersect = dot(start - v2, n) / (para23 ? 0.000001 : dot(l23, n));
	float d31intersect = dot(start - v3, n) / (para31 ? 0.000001 : dot(l31, n));
	bool hasIntersect12 = (!para12) && d12intersect > 0 && d12intersect < dis12;
	bool hasIntersect23 = (!para23) && d23intersect > 0 && d23intersect < dis23;
	bool hasIntersect31 = (!para31) && d31intersect > 0 && d31intersect < dis31;


	int separateVectex, bottomV1, bottomV2;
	float3 intersect1, intersect2, disturb = 0.0001*n;
	float3 intersectNorm1, intersectNorm2; //temporary solution for norms
	//assume it is impossible that hasIntersect12 && hasIntersect23 && hasIntersect31
	if (hasIntersect12 && hasIntersect23){
		separateVectex = inds.y;// separateVectex = 2;
		if (dot(v2 - start, n) < 0) disturb = -disturb;
		bottomV1 = inds.x;
		bottomV2 = inds.z;
		intersect1 = v1 + d12intersect * l12;
		intersect2 = v2 + d23intersect * l23;

		intersectNorm1 = normalize((norm2 * d12intersect + norm1 * (dis12 - d12intersect)) / dis12);
		intersectNorm2 = normalize((norm3 * d23intersect + norm2 * (dis23 - d23intersect)) / dis23);
	}
	else if (hasIntersect23 && hasIntersect31){
		separateVectex = inds.z; // separateVectex = 3;
		if (dot(v3 - start, n) < 0) disturb = -disturb;
		bottomV1 = inds.y;
		bottomV2 = inds.x;
		intersect1 = v2 + d23intersect * l23;
		intersect2 = v3 + d31intersect * l31;

		intersectNorm1 = normalize((norm3 * d23intersect + norm2 * (dis23 - d23intersect)) / dis23);
		intersectNorm2 = normalize((norm1 * d31intersect + norm3 * (dis31 - d31intersect)) / dis31);
	}
	else if (hasIntersect31 && hasIntersect12){
		separateVectex = inds.x; //separateVectex = 1;
		if (dot(v1 - start, n) < 0) disturb = -disturb;
		bottomV1 = inds.z;
		bottomV2 = inds.y;
		intersect1 = v3 + d31intersect * l31;
		intersect2 = v1 + d12intersect * l12;

		intersectNorm1 = normalize((norm1 * d31intersect + norm3 * (dis31 - d31intersect)) / dis31);
		intersectNorm2 = normalize((norm2 * d12intersect + norm1 * (dis12 - d12intersect)) / dis12);
	}
	//NOTE!!! one case is now missing. it is possible that only one of the three booleans is true
	else{
		return;
	}

	float projLength1long = dot(intersect1 - start, tunnelVec);
	float projLength1short = dot(intersect1 - start, dir2nd);
	float projLength2long = dot(intersect2 - start, tunnelVec);
	float projLength2short = dot(intersect2 - start, dir2nd);
	if ((projLength1long > 0 && projLength1long < tunnelLength && abs(projLength1short) < deformationScaleVertical)
		|| (projLength2long > 0 && projLength2long < tunnelLength && abs(projLength2short) < deformationScaleVertical)){
		indices[3 * i] = 0;
		indices[3 * i + 1] = 0;
		indices[3 * i + 2] = 0;

		int numAddedFacesBefore = atomicAdd(numAddedFaces, 3); //each divided triangle creates 3 new faces

		int curNumVertex = vertexcount + 4 * numAddedFacesBefore / 3; //each divided triangle creates 4 new vertex
		vertexCoords[3 * curNumVertex] = intersect1.x + disturb.x;
		vertexCoords[3 * curNumVertex + 1] = intersect1.y + disturb.y;
		vertexCoords[3 * curNumVertex + 2] = intersect1.z + disturb.z;
		vertexCoords[3 * (curNumVertex + 1)] = intersect2.x + disturb.x;
		vertexCoords[3 * (curNumVertex + 1) + 1] = intersect2.y + disturb.y;
		vertexCoords[3 * (curNumVertex + 1) + 2] = intersect2.z + disturb.z;
		vertexCoords[3 * (curNumVertex + 2)] = intersect1.x - disturb.x;
		vertexCoords[3 * (curNumVertex + 2) + 1] = intersect1.y - disturb.y;
		vertexCoords[3 * (curNumVertex + 2) + 2] = intersect1.z - disturb.z;
		vertexCoords[3 * (curNumVertex + 3)] = intersect2.x - disturb.x;
		vertexCoords[3 * (curNumVertex + 3) + 1] = intersect2.y - disturb.y;
		vertexCoords[3 * (curNumVertex + 3) + 2] = intersect2.z - disturb.z;


		vertexColorVals[curNumVertex] = vertexColorVals[separateVectex];
		vertexColorVals[curNumVertex + 1] = vertexColorVals[separateVectex];
		vertexColorVals[curNumVertex + 2] = vertexColorVals[separateVectex];
		vertexColorVals[curNumVertex + 3] = vertexColorVals[separateVectex];

		norms[3 * curNumVertex] = intersectNorm1.x;
		norms[3 * curNumVertex + 1] = intersectNorm1.y;
		norms[3 * curNumVertex + 2] = intersectNorm1.z;
		norms[3 * (curNumVertex + 1)] = intersectNorm2.x;
		norms[3 * (curNumVertex + 1) + 1] = intersectNorm2.y;
		norms[3 * (curNumVertex + 1) + 2] = intersectNorm2.z;
		norms[3 * (curNumVertex + 2)] = intersectNorm1.x;
		norms[3 * (curNumVertex + 2) + 1] = intersectNorm1.y;
		norms[3 * (curNumVertex + 2) + 2] = intersectNorm1.z;
		norms[3 * (curNumVertex + 3)] = intersectNorm2.x;
		norms[3 * (curNumVertex + 3) + 1] = intersectNorm2.y;
		norms[3 * (curNumVertex + 3) + 2] = intersectNorm2.z;


		int curNumFaces = numAddedFacesBefore + facecount;

		indices[3 * curNumFaces] = separateVectex;
		indices[3 * curNumFaces + 1] = curNumVertex + 1;  //order of vertex matters! use counter clockwise
		indices[3 * curNumFaces + 2] = curNumVertex;
		indices[3 * (curNumFaces + 1)] = bottomV1;
		indices[3 * (curNumFaces + 1) + 1] = curNumVertex + 2;
		indices[3 * (curNumFaces + 1) + 2] = curNumVertex + 3;
		indices[3 * (curNumFaces + 2)] = bottomV2;
		indices[3 * (curNumFaces + 2) + 1] = bottomV1;
		indices[3 * (curNumFaces + 2) + 2] = curNumVertex + 3;
	}
	else {
		return;
	}

}


void PositionBasedDeformProcessor::modifyPolyMesh()
{
	cudaMemcpy(d_vertexCoords, poly->vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, poly->indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_norms, poly->vertexNorms, sizeof(float)*poly->vertexcount * 3, cudaMemcpyHostToDevice);

	cudaMemset(d_vertexDeviateVals, 0, sizeof(float)*poly->vertexcount * 2);
	cudaMemcpy(d_vertexColorVals, poly->vertexColorVals, sizeof(float)*poly->vertexcount, cudaMemcpyHostToDevice);

	cudaMemset(d_numAddedFaces, 0, sizeof(int));


	int threadsPerBlock = 64;
	int blocksPerGrid = (poly->vertexcount + threadsPerBlock - 1) / threadsPerBlock;

	d_disturbVertex << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, poly->vertexcount,
		tunnelStart, tunnelEnd, deformationScaleVertical, rectVerticalDir);

	threadsPerBlock = 64;
	blocksPerGrid = (poly->facecount + threadsPerBlock - 1) / threadsPerBlock;
	d_modifyMesh << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, d_indices, poly->facecountOri, poly->vertexcountOri, d_norms,
		tunnelStart, tunnelEnd, deformationScale, deformationScale, deformationScaleVertical, rectVerticalDir,
		d_numAddedFaces, d_vertexColorVals);

	int numAddedFaces;
	cudaMemcpy(&numAddedFaces, d_numAddedFaces, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << "added new face count " << numAddedFaces << std::endl;

	poly->facecount += numAddedFaces;
	poly->vertexcount += numAddedFaces / 3 * 4;

	std::cout << "old face count " << poly->facecountOri << std::endl;
	std::cout << "new face count " << poly->facecount << std::endl;
	std::cout << "old vertex count " << poly->vertexcountOri << std::endl;
	std::cout << "new vertex count " << poly->vertexcount << std::endl;

	cudaMemcpy(poly->indices, d_indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(poly->vertexCoords, d_vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(poly->vertexNorms, d_norms, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(poly->vertexColorVals, d_vertexColorVals, sizeof(float)*poly->vertexcount, cudaMemcpyDeviceToHost);

	cudaMemcpy(d_vertexCoords_init, d_vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);
}


bool PositionBasedDeformProcessor::inRange(float3 v)
{
	if (dataType == VOLUME){
		v = v / volume->spacing;
	}
	return v.x >= minPos.x && v.x < maxPos.x && v.y >= minPos.y && v.y < maxPos.y &&v.z >= minPos.z && v.z < maxPos.z;
}

void PositionBasedDeformProcessor::deformDataByDegree(float r)
{
	if (dataType == VOLUME){
		doVolumeDeform(r);
	}
	else if (dataType == MESH){
		doPolyDeform(r);
	}
	else if (dataType == PARTICLE){
		doParticleDeform(r);
	}
	else{
		std::cout << " inRange not implemented " << std::endl;
		exit(0);
	}
}

void PositionBasedDeformProcessor::deformDataByDegree2Tunnel(float r, float rClose)
{
	if (dataType == VOLUME){
		doVolumeDeform2Tunnel(r, rClose);
	}
	else if (dataType == MESH){
	}
	else if (dataType == PARTICLE){
	}
	else{
		std::cout << " inRange not implemented " << std::endl;
		exit(0);
	}

	return;
}

void PositionBasedDeformProcessor::resetData()
{
	if (dataType == VOLUME){
		volume->reset();
	}
	else if (dataType == MESH){
		poly->reset();
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


bool PositionBasedDeformProcessor::process(float* modelview, float* projection, int winWidth, int winHeight)
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

			if (dataType == MESH){ //for poly data, the original data will be modified, which is not applicable to other types of data
				modifyPolyMesh();
			}

			//start a opening animation
			hasOpenAnimeStarted = true;
			hasCloseAnimeStarted = false; //currently if there is closing procedure for other tunnels, they are finished suddenly
			startOpen = std::clock();
			//}
			//else if (lastEyeState == inWall){
			//from wall to wall
			//}
		}
		else if (inRange(eyeInLocal) && !atProperLocationInOriData(eyeInLocal)){	// in solid area
			// in this case, set the start of deformation
			if (lastEyeState != inWall){
				lastDataState = DEFORMED;
				lastEyeState = inWall;

				computeTunnelInfo(eyeInLocal);

				if (dataType == MESH){ //for poly data, the original data will be modified, which is not applicable to other types of data
					modifyPolyMesh();
				}

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
			// either eyeInLocal is out of range, or eyeInLocal is at proper location
			//in this case, no state change
		}
	}
	else{ //lastDataState == Deformed
		if (isForceDeform){

		}
		else if (inRange(eyeInLocal) && !atProperLocationInOriData(eyeInLocal)){
			//in area which is solid in the original volume
			bool atProper = atProperLocationInDeformedData(eyeInLocal);
			if (atProper){
				// not in the solid region in the deformed volume
				// in this case, no change
			}
			else{
				//std::cout <<"Triggered "<< lastDataState << " " << lastEyeState << " " << hasOpenAnimeStarted << " " << hasCloseAnimeStarted << std::endl;
				//even in the deformed volume, eye is still inside the solid region 
				//eye should just move to a solid region


				sdkResetTimer(&timer);
				sdkStartTimer(&timer);

				sdkResetTimer(&timerFrame);

				fpsCount = 0;

				lastOpenFinalDegree = closeStartingRadius;
				lastDeformationDirVertical = rectVerticalDir;
				lastTunnelStart = tunnelStart;
				lastTunnelEnd = tunnelEnd;

				computeTunnelInfo(eyeInLocal);

				hasOpenAnimeStarted = true;//start a opening animation
				hasCloseAnimeStarted = true; //since eye should just moved to the current solid, the previous solid should be closed 
				startOpen = std::clock();
			}
		}
		else{// in area which is proper in the original volume
			hasCloseAnimeStarted = true;
			hasOpenAnimeStarted = false;
			startClose = std::clock();

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

		std::cout << "doing openning with r: " << r << std::endl;

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


void PositionBasedDeformProcessor::InitCudaSupplies()
{
	volumeTexInput.normalized = false;
	volumeTexInput.filterMode = cudaFilterModeLinear;
	volumeTexInput.addressMode[0] = cudaAddressModeBorder;
	volumeTexInput.addressMode[1] = cudaAddressModeBorder;
	volumeTexInput.addressMode[2] = cudaAddressModeBorder;

	volumePointTexture.normalized = false;
	volumePointTexture.filterMode = cudaFilterModePoint;
	volumePointTexture.addressMode[0] = cudaAddressModeBorder;
	volumePointTexture.addressMode[1] = cudaAddressModeBorder;
	volumePointTexture.addressMode[2] = cudaAddressModeBorder;

	if (volume != 0){
		//currently only for volume data
		volumeCudaIntermediate = std::make_shared<VolumeCUDA>();
		volumeCudaIntermediate->VolumeCUDA_deinit();
		volumeCudaIntermediate->VolumeCUDA_init(volume->size, volume->values, 1, 1);
		//volumeCudaIntermediate.VolumeCUDA_init(volume->size, 0, 1, 1);//??
	}


	transferTex2.normalized = true;
	transferTex2.filterMode = cudaFilterModeLinear;
	transferTex2.addressMode[0] = cudaAddressModeClamp;
}




