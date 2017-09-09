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

#define RESOLU 5

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
	PrepareDataStructureForVolumeDeform();
	sdkCreateTimer(&timer);
	sdkCreateTimer(&timerFrame);
	minPos = make_float3(0, 0, 0);
	maxPos = make_float3(volume->size.x, volume->size.y, volume->size.z);
	dataType = VOLUME;
};


void PositionBasedDeformProcessor::PrepareDataStructureForVolumeDeform()
{
	volumeTexInput.normalized = false;
	volumeTexInput.filterMode = cudaFilterModeLinear;	//for deformation
	volumeTexInput.addressMode[0] = cudaAddressModeBorder;
	volumeTexInput.addressMode[1] = cudaAddressModeBorder;
	volumeTexInput.addressMode[2] = cudaAddressModeBorder;

	volumePointTexture.normalized = false;
	volumePointTexture.filterMode = cudaFilterModePoint;	//for detection of proper location
	volumePointTexture.addressMode[0] = cudaAddressModeBorder;
	volumePointTexture.addressMode[1] = cudaAddressModeBorder;
	volumePointTexture.addressMode[2] = cudaAddressModeBorder;

	volumeCudaIntermediate = std::make_shared<VolumeCUDA>();
	volumeCudaIntermediate->VolumeCUDA_deinit();
	volumeCudaIntermediate->VolumeCUDA_init(volume->size, volume->values, 1, 1);

	transferTex2.normalized = true;
	transferTex2.filterMode = cudaFilterModeLinear;
	transferTex2.addressMode[0] = cudaAddressModeClamp;
}


PositionBasedDeformProcessor::PositionBasedDeformProcessor(std::shared_ptr<PolyMesh> ori, std::shared_ptr<MatrixManager> _m)
{
	poly = ori;
	matrixMgr = _m;
	PrepareDataStructureForPolyDeform();
	sdkCreateTimer(&timer);
	sdkCreateTimer(&timerFrame);
	dataType = MESH;
	//minPos and maxPos need to be set externally

	initIntersectionInfoForCircleAndPoly();
};

void PositionBasedDeformProcessor::PrepareDataStructureForPolyDeform()
{
	if (d_vertexCoords) { cudaFree(d_vertexCoords); d_vertexCoords = 0; };
	if (d_vertexCoords_init){ cudaFree(d_vertexCoords_init); d_vertexCoords_init = 0; };
	if (d_indices){ cudaFree(d_indices); d_indices = 0; };
	if (d_norms){ cudaFree(d_norms); d_norms = 0; };
	if (d_vertexDeviateVals){ cudaFree(d_vertexDeviateVals); d_vertexDeviateVals = 0; };
	if (d_vertexColorVals) { cudaFree(d_vertexColorVals); d_vertexColorVals = 0; };
	if (d_numAddedFaces){ cudaFree(d_numAddedFaces); d_numAddedFaces = 0; };

	//NOTE!! here doubled the space. Hopefully it is large enough
	cudaMalloc(&d_vertexCoords, sizeof(float)*poly->vertexcount * 3 * 2);
	cudaMalloc(&d_vertexCoords_init, sizeof(float)*poly->vertexcount * 3 * 2);
	cudaMalloc(&d_indices, sizeof(unsigned int)*poly->facecount * 3 * 2);
	cudaMalloc(&d_norms, sizeof(float)*poly->vertexcount * 3 * 2);
	cudaMalloc(&d_vertexDeviateVals, sizeof(float)*poly->vertexcount * 2);
	cudaMalloc(&d_vertexColorVals, sizeof(float)*poly->vertexcount * 2);
	cudaMalloc(&d_numAddedFaces, sizeof(int));

	cudaMemcpy(d_vertexCoords_init, poly->vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, poly->indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyHostToDevice);
}

void PositionBasedDeformProcessor::polyMeshDataUpdated()
{
	PrepareDataStructureForPolyDeform();
}


PositionBasedDeformProcessor::PositionBasedDeformProcessor(std::shared_ptr<Particle> ori, std::shared_ptr<MatrixManager> _m)
{
	particle = ori;
	matrixMgr = _m;

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

	if (systemState == DEFORMED && isActive){
		doParticleDeform(r);
	}
	else{
		d_vec_posTarget.assign(&(particle->pos[0]), &(particle->pos[0]) + particle->numParticles);
	}
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
		std::cout << " resetData not implemented " << std::endl;
		exit(0);
	}
}


//////////////////////tunnel related////////////////////////////

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
		while (atProperLocation(tunnelStart, true)){
			tunnelStart += tunnelAxis*step;
		}
		tunnelEnd = tunnelStart + tunnelAxis*step;
		while (!atProperLocation(tunnelEnd, true)){
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
		while (!atProperLocation(tunnelEnd, true)){
			tunnelEnd += tunnelAxis*step;
		}
		tunnelStart = centerPoint;
		while (!atProperLocation(tunnelStart, true)){
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

bool PositionBasedDeformProcessor::sameTunnel(){
	float thr = 0.0001;
	if (shapeModel == CUBOID){
		return length(lastTunnelStart - tunnelStart) < thr	&& length(lastTunnelEnd - tunnelEnd) < thr && length(lastDeformationDirVertical - rectVerticalDir) < thr;
	}
	if (shapeModel == CIRCLE){
		return length(lastTunnelStart - tunnelStart) < thr	&& length(lastTunnelEnd - tunnelEnd) < thr;
	}
	else{
		std::cout << " sameTunnel not implemented " << std::endl;
		return false;
	}
}



//////////////////////proper location detection////////////////////////////
bool PositionBasedDeformProcessor::inRange(float3 v)
{
	if (dataType == VOLUME){
		v = v / volume->spacing;
	}
	return v.x >= minPos.x && v.x < maxPos.x && v.y >= minPos.y && v.y < maxPos.y &&v.z >= minPos.z && v.z < maxPos.z;
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

	//float dis = min(min(length(pos - v1), length(pos - v2)), length(pos - v3));
	float dis = d_disToTri(pos, v1, v2, v3, thr);
	if (dis < thr)
	{
		*res = true;
	}

	return;
}

bool PositionBasedDeformProcessor::atProperLocation(float3 pos, bool useOriData)
{
	if (!inRange(pos)){
		return true;
	}

	if (!useOriData){
		if (systemState != ORIGINAL && systemState != CLOSING){
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
	}

	if (dataType == VOLUME){
		cudaChannelFormatDesc channelFloat4 = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaBindTextureToArray(transferTex2, rcp->d_transferFunc, channelFloat4));

		bool* d_atProper;
		cudaMalloc(&d_atProper, sizeof(bool)* 1);
		cudaChannelFormatDesc cd2 = volume->volumeCudaOri.channelDesc;
		if (useOriData){
			checkCudaErrors(cudaBindTextureToArray(volumePointTexture, volume->volumeCudaOri.content, cd2));
		}
		else{
			checkCudaErrors(cudaBindTextureToArray(volumePointTexture, volume->volumeCuda.content, cd2));
		}
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
		if (useOriData){
			d_checkIfTooCloseToPoly << <blocksPerGrid, threadsPerBlock >> >(pos, d_indices, poly->facecount, d_vertexCoords_init, disThr, d_tooCloseToData);
		}
		else{
			d_checkIfTooCloseToPoly << <blocksPerGrid, threadsPerBlock >> >(pos, d_indices, poly->facecount, d_vertexCoords, disThr, d_tooCloseToData);
		}

	/*	std::vector<float> aaa(poly->facecount, 0);
		cudaMemcpy(&(aaa[0]), d_dis, sizeof(float)*  poly->facecount, cudaMemcpyDeviceToHost);
*/

		bool tooCloseToData;
		cudaMemcpy(&tooCloseToData, d_tooCloseToData, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		cudaFree(d_tooCloseToData);
		return !tooCloseToData;
	}
	else if (dataType == PARTICLE){
		float init = 10000;
		float inSavePosition;
		if (useOriData){
			inSavePosition = thrust::transform_reduce(
				d_vec_posOrig.begin(), d_vec_posOrig.end(), functor_dis(pos), init, thrust::minimum<float>());
		}
		else{
			inSavePosition = thrust::transform_reduce(
				d_vec_posTarget.begin(), d_vec_posTarget.end(), functor_dis(pos), init, thrust::minimum<float>());
		}
		return (inSavePosition > disThr);
	}
	else{
		std::cout << " in data not implemented " << std::endl;
		exit(0);
	}


}



//////////////////////cut mesh////////////////////////////

__device__ inline float3 d_disturbVertexSingle_circle(float* vertexCoords, float3 start, float3 end, float3 pos)
//if a vertex is too close to the cutting plane, then disturb it a little to avoid numerical error
{
	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);

	float thr = 0.000001;
	float disturb = 0.00001;

	float3 voxelVec = pos - start;
	float l = dot(voxelVec, tunnelVec);
	if (l > 0 && l < tunnelLength){
		float3 prjPoint = start + l*tunnelVec;
		float dis = length(pos - prjPoint);

		//when dis==0 , disturb the vertex a little to avoid numerical error
		if (dis < thr){
			float3 ref = make_float3(0, 0, 1);
			float3 disturbVec;
			if (abs(dot(ref, tunnelVec)) < 0.9){
				disturbVec = normalize(cross(ref, tunnelVec))*disturb;
			}
			else{
				disturbVec = normalize(cross(make_float3(0, 1, 0), tunnelVec))*disturb;
			}

			pos += disturbVec;

			*vertexCoords = pos.x;
			*(vertexCoords + 1) = pos.y;
			*(vertexCoords + 2) = pos.z;
		}
	}
	return pos;
}

__device__ inline void setNew(float* vertexCoords, float3 pos)
{
	*vertexCoords = pos.x;
	*(vertexCoords + 1) = pos.y;
	*(vertexCoords + 2) = pos.z;
}

__global__ void d_disturbVertex_CircleModel(float* vertexCoords, unsigned int* indices, int facecount, float3 start, float3 end)
//if a vertex or an egde is too close to the cutting axis, then disturb it a little to avoid numerical error
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= facecount)	return;
	uint3 inds = make_uint3(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);
	float3 v1 = make_float3(vertexCoords[3 * inds.x], vertexCoords[3 * inds.x + 1], vertexCoords[3 * inds.x + 2]);
	float3 v2 = make_float3(vertexCoords[3 * inds.y], vertexCoords[3 * inds.y + 1], vertexCoords[3 * inds.y + 2]);
	float3 v3 = make_float3(vertexCoords[3 * inds.z], vertexCoords[3 * inds.z + 1], vertexCoords[3 * inds.z + 2]);
	
	float thr = 0.000001;
	float disturb = 0.00001;
	float3 tunnelVec = normalize(end - start);

	v1 = d_disturbVertexSingle_circle(vertexCoords + 3 * inds.x, start, end, v1);
	v2 = d_disturbVertexSingle_circle(vertexCoords + 3 * inds.y, start, end, v2);
	v3 = d_disturbVertexSingle_circle(vertexCoords + 3 * inds.z, start, end, v3);

	//suppose any 2 points of the triangle are not overlapping
	float dis12 = length(v2 - v1);
	float3 l12 = normalize(v2 - v1);
	float dis23 = length(v3 - v2);
	float3 l23 = normalize(v3 - v2);
	float dis31 = length(v1 - v3);
	float3 l31 = normalize(v1 - v3);

	//https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
	float3 triNormal = normalize(cross(l12, l31));
	bool isPara = abs(dot(tunnelVec, triNormal)) < 0.000001;
	float dis_startToIntersectionPoint = dot(v1 - start, triNormal) / (isPara ? 0.000001 : dot(tunnelVec, triNormal));
	bool hasIntersect = (!isPara) && dis_startToIntersectionPoint > 0 && dis_startToIntersectionPoint < length(end - start);
	if (!hasIntersect){
		return;
	}

	float3 intersect = start + dis_startToIntersectionPoint * tunnelVec;
	bool isInside = false;
	if (dot(cross(l12, -l31), cross(l12, intersect - v1)) >= 0){
		if (dot(cross(l23, -l12), cross(l23, intersect - v2)) >= 0){
			if (dot(cross(l31, -l23), cross(l31, intersect - v3)) >= 0){
				isInside = true;
			}
		}
	}
	if (!isInside){
		return;
	}

	{
		float3 prjOn12 = v1 + dot(intersect - v1, l12);
		float disTo12 = length(intersect - prjOn12);
		if (disTo12 == 0){
			float3 disturbVec;
			if (abs(dot(make_float3(0, 0, 1), tunnelVec)) < 0.9){
				disturbVec = normalize(cross(make_float3(0, 0, 1), tunnelVec))*disturb;
			}
			else{
				disturbVec = normalize(cross(make_float3(0, 1, 0), tunnelVec))*disturb;
			}
			setNew(vertexCoords + 3 * inds.x, v1 + disturbVec);
			setNew(vertexCoords + 3 * inds.y, v2 + disturbVec);
			return;
		}
		else if (disTo12 < thr){
			float3 disturbVec = normalize(prjOn12 - intersect)*disturb;
			setNew(vertexCoords + 3 * inds.x, v1 + disturbVec);
			setNew(vertexCoords + 3 * inds.y, v2 + disturbVec);
			return;
		}
	}

	{
		float3 prjOn23 = v2 + dot(intersect - v2, l23);
		float disTo23 = length(intersect - prjOn23);
		if (disTo23 == 0){
			float3 disturbVec;
			if (abs(dot(make_float3(0, 0, 1), tunnelVec)) < 0.9){
				disturbVec = normalize(cross(make_float3(0, 0, 1), tunnelVec))*disturb;
			}
			else{
				disturbVec = normalize(cross(make_float3(0, 1, 0), tunnelVec))*disturb;
			}
			setNew(vertexCoords + 3 * inds.y, v2 + disturbVec);
			setNew(vertexCoords + 3 * inds.z, v3 + disturbVec);
			return;
		}
		else if (disTo23 < thr){
			float3 disturbVec = normalize(prjOn23 - intersect)*disturb;
			setNew(vertexCoords + 3 * inds.y, v2 + disturbVec);
			setNew(vertexCoords + 3 * inds.z, v3 + disturbVec);
			return;
		}
	}

	{
		float3 prjOn31 = v3 + dot(intersect - v3, l31);
		float disTo31 = length(intersect - prjOn31);
		if (disTo31 == 0){
			float3 disturbVec;
			if (abs(dot(make_float3(0, 0, 1), tunnelVec)) < 0.9){
				disturbVec = normalize(cross(make_float3(0, 0, 1), tunnelVec))*disturb;
			}
			else{
				disturbVec = normalize(cross(make_float3(0, 1, 0), tunnelVec))*disturb;
			}
			setNew(vertexCoords + 3 * inds.z, v3 + disturbVec);
			setNew(vertexCoords + 3 * inds.x, v1 + disturbVec);
			return;
		}
		else if (disTo31 < thr){
			float3 disturbVec = normalize(prjOn31 - intersect)*disturb;
			setNew(vertexCoords + 3 * inds.z, v3 + disturbVec);
			setNew(vertexCoords + 3 * inds.x, v1 + disturbVec);
			return;
		}
	}

	return;
}

__global__ void d_modifyMeshKernel_CircledModel_round1(float* vertexCoords, unsigned int* indices, int facecount, float3 start, float3 end, int* numIntersectTris, unsigned int* intersectedTris)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= facecount)	return;

	uint3 inds = make_uint3(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);
	float3 v1 = make_float3(vertexCoords[3 * inds.x], vertexCoords[3 * inds.x + 1], vertexCoords[3 * inds.x + 2]);
	float3 v2 = make_float3(vertexCoords[3 * inds.y], vertexCoords[3 * inds.y + 1], vertexCoords[3 * inds.y + 2]);
	float3 v3 = make_float3(vertexCoords[3 * inds.z], vertexCoords[3 * inds.z + 1], vertexCoords[3 * inds.z + 2]);

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
	float3 triNormal = normalize(cross(l12, l31));
	bool isPara = abs(dot(tunnelVec, triNormal)) < 0.000001;
	float dis_startToIntersectionPoint = dot(v1 - start, triNormal) / (isPara ? 0.000001 : dot(tunnelVec, triNormal));
	bool hasIntersect = (!isPara) && dis_startToIntersectionPoint > 0 && dis_startToIntersectionPoint < length(end - start);
	if (!hasIntersect){
		return;
	}

	float3 intersect = start + dis_startToIntersectionPoint * tunnelVec;
	bool isInside = false;
	if (dot(cross(l12, -l31), cross(l12, intersect - v1)) >= 0){
		if (dot(cross(l23, -l12), cross(l23, intersect - v2)) >= 0){
			if (dot(cross(l31, -l23), cross(l31, intersect - v3)) >= 0){
				isInside = true;
			}
		}
	}
	if (!isInside){
		return;
	}

	int numIntersectTriBefore = atomicAdd(numIntersectTris, 1);
	if (numIntersectTriBefore < MAX_CIRCLE_INTERACT)
		intersectedTris[numIntersectTriBefore] = i;
}

__global__ void d_modifyMeshKernel_CircledModel_round2(unsigned int* indices, int facecount, float3 start, float3 end, int numIntersectTris, unsigned int* intersectedTris, int* neighborIdsOfIntersectedTris)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= facecount)	return;

	uint3 inds = make_uint3(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);

	bool isNeighbor = false;
	for (int j = 0; j < numIntersectTris; j++){
		int intersectTriId = intersectedTris[j];
		if (i == intersectTriId){
			return;
		}
	}
	for (int j = 0; j < numIntersectTris; j++){
		int intersectTriId = intersectedTris[j];
		uint3 curInds = make_uint3(indices[3 * intersectTriId], indices[3 * intersectTriId + 1], indices[3 * intersectTriId + 2]);
		if ((inds.x == curInds.x && inds.y == curInds.y) || (inds.y == curInds.x && inds.x == curInds.y)){//xy->xy
			neighborIdsOfIntersectedTris[3 * j] = inds.z;
			isNeighbor = true;
		}
		else if ((inds.x == curInds.x && inds.z == curInds.y) || (inds.z == curInds.x && inds.x == curInds.y)){//xz->xy
			neighborIdsOfIntersectedTris[3 * j] = inds.y;
			isNeighbor = true;
		}
		else if ((inds.y == curInds.x && inds.z == curInds.y) || (inds.z == curInds.x && inds.y == curInds.y)){//yz->xy
			neighborIdsOfIntersectedTris[3 * j] = inds.x;
			isNeighbor = true;
		}
		else 		
		if ((inds.x == curInds.y && inds.y == curInds.z) || (inds.y == curInds.y && inds.x == curInds.z)){//xy->yz
			neighborIdsOfIntersectedTris[3 * j + 1] = inds.z;
			isNeighbor = true;
		}
		else if ((inds.x == curInds.y && inds.z == curInds.z) || (inds.z == curInds.y && inds.x == curInds.z)){//xz->yz
			neighborIdsOfIntersectedTris[3 * j + 1] = inds.y;
			isNeighbor = true;
		}
		else if ((inds.y == curInds.y && inds.z == curInds.z) || (inds.z == curInds.y && inds.y == curInds.z)){//yz->yz
			neighborIdsOfIntersectedTris[3 * j + 1] = inds.x;
			isNeighbor = true;
		}
		else
		if ((inds.x == curInds.z && inds.y == curInds.x) || (inds.y == curInds.z && inds.x == curInds.x)){//xy->zx
			neighborIdsOfIntersectedTris[3 * j + 2] = inds.z;
			isNeighbor = true;
		}
		else if ((inds.x == curInds.z && inds.z == curInds.x) || (inds.z == curInds.z && inds.x == curInds.x)){//xz->zx
			neighborIdsOfIntersectedTris[3 * j + 2] = inds.y;
			isNeighbor = true;
		}
		else if ((inds.y == curInds.z && inds.z == curInds.x) || (inds.z == curInds.z && inds.y == curInds.x)){//yz->zx
			neighborIdsOfIntersectedTris[3 * j + 2] = inds.x;
			isNeighbor = true;
		}
	}
	if (isNeighbor)
	{
		//erase current triangle
		indices[3 * i] = 0;
		indices[3 * i + 1] = 0;
		indices[3 * i + 2] = 0;
	}
}

__global__ void d_modifyMeshKernel_CircledModel_round3(float* vertexCoords, unsigned int* indices, int facecount, int vertexcount, float* norms, float3 start, float3 end, int* numAddedFaces, int* numAddedVertices, float* vertexColorVals, unsigned int* intersectedTris, int* neighborIdsOfIntersectedTris, int numIntersectTris)
{
	//now the point is inside of the triangle. Due to the distrubance, it is supposed to be far enough from the 3 edges and vertices of the triangle

	int indThread = blockDim.x * blockIdx.x + threadIdx.x;
	if (indThread >= numIntersectTris)	return;

	unsigned int i = intersectedTris[indThread];

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
	float3 triNormal = normalize(cross(l12, l31));
	bool isPara = abs(dot(tunnelVec, triNormal)) < 0.000001;
	float dis_startToIntersectionPoint = dot(v1 - start, triNormal) / (isPara ? 0.000001 : dot(tunnelVec, triNormal));
	bool hasIntersect = (!isPara) && dis_startToIntersectionPoint > 0 && dis_startToIntersectionPoint < length(end - start);
	if (!hasIntersect){
		return;  //should never return
	}

	float3 intersect = start + dis_startToIntersectionPoint * tunnelVec;
	/*
	bool isInside = false;
	if (dot(cross(l12, -l31), cross(l12, intersect - v1)) >= 0){
	if (dot(cross(l23, -l12), cross(l23, intersect - v2)) >= 0){
	if (dot(cross(l31, -l23), cross(l31, intersect - v3)) >= 0){
	isInside = true;
	}
	}
	}
	if (!isInside){
	return; //should never return
	}
	*/


	int numNewVerticesEachSide = RESOLU + 1 + RESOLU;//1 for small triangle; RESOLU on the arc; RESOLU on the original edge
	int numAddedVerticesBefore = atomicAdd(numAddedVertices, numNewVerticesEachSide * 3);
	int curNumVertex = vertexcount + numAddedVerticesBefore;

	float disto1 = length(intersect - v1);
	float disto2 = length(intersect - v2);
	float disto3 = length(intersect - v3);
	float disAllDouble = 2 * (disto1 + disto2 + disto3);
	float3 newNormal = (disto2 + disto3) / disAllDouble*norm1 + (disto1 + disto3) / disAllDouble*norm2 + (disto2 + disto1) / disAllDouble*norm3; //simple interpolation
	float newColorVal = vertexColorVals[inds.x];//any one of the 3 vertices. assume the 3 values are the same for a triangle


	float thr = 0.000005;
	float3 vNew[3] = { intersect + normalize(v1 - intersect)*thr, intersect + normalize(v2 - intersect)*thr, intersect + normalize(v3 - intersect)*thr };
	for (int j = 0; j < 3; j++){
		vertexCoords[3 * (curNumVertex + j * numNewVerticesEachSide)] = vNew[j].x;
		vertexCoords[3 * (curNumVertex + j* numNewVerticesEachSide) + 1] = vNew[j].y;
		vertexCoords[3 * (curNumVertex + j* numNewVerticesEachSide) + 2] = vNew[j].z;
		norms[3 * (curNumVertex + j * numNewVerticesEachSide)] = newNormal.x;
		norms[3 * (curNumVertex + j * numNewVerticesEachSide) + 1] = newNormal.y;
		norms[3 * (curNumVertex + j * numNewVerticesEachSide) + 2] = newNormal.z;
		vertexColorVals[curNumVertex + j * numNewVerticesEachSide] = newColorVal;
	}
	float3 vOld[3] = { v1, v2, v3 };
	float3 normOld[3] = { norm1, norm2, norm3 };
	for (int j = 0; j < 3; j++){
		for (int jj = 1; jj <= RESOLU; jj++){
			float ratio1 = 1.0 * jj / (RESOLU + 1.0);

			int curId = curNumVertex + j * numNewVerticesEachSide + 1 + (jj-1) * 2;
			float3 temp1 = vNew[j] * (1 - ratio1) + vNew[(j + 1) % 3] * ratio1;
			temp1 = intersect + normalize(temp1 - intersect)*thr;
			vertexCoords[3 * curId] = temp1.x;
			vertexCoords[3 * curId + 1] = temp1.y;
			vertexCoords[3 * curId + 2] = temp1.z;
			norms[3 * curId] = newNormal.x;
			norms[3 * curId + 1] = newNormal.y;
			norms[3 * curId + 2] = newNormal.z;
			vertexColorVals[curId] = newColorVal;

			curId++;
			temp1 = vOld[j] * (1 - ratio1) + vOld[(j + 1) % 3] * ratio1;
			float3 sideNormal = normOld[j] * (1 - ratio1) + normOld[(j + 1) % 3] * ratio1;
			vertexCoords[3 * curId] = temp1.x;
			vertexCoords[3 * curId + 1] = temp1.y;
			vertexCoords[3 * curId + 2] = temp1.z;
			norms[3 * curId] = sideNormal.x;
			norms[3 * curId + 1] = sideNormal.y;
			norms[3 * curId + 2] = sideNormal.z;
			vertexColorVals[curId] = newColorVal;
		}
	}

	int neighborIds[3] = {neighborIdsOfIntersectedTris[3 * indThread], neighborIdsOfIntersectedTris[3 * indThread + 1], neighborIdsOfIntersectedTris[3 * indThread + 2]};

	int numNewFaces = (RESOLU + 1) * 2 * 3 + (((neighborIds[0] > -1) ? 1 : 0) + ((neighborIds[1] > -1) ? 1 : 0) + ((neighborIds[2] > -1) ? 1 : 0)) * (RESOLU + 1);
	
	int numAddedFacesBefore = atomicAdd(numAddedFaces, numNewFaces);
	int curNumFaces = numAddedFacesBefore + facecount;
	
	uint vOldId[3] = { inds.x, inds.y, inds.z };

	for (int j = 0; j < 3; j++){
		for (int jj = 0; jj <= RESOLU; jj++){
			int faceid = curNumFaces + j * (RESOLU + 1) * 2 + jj * 2;
			if (jj == 0){
				int startVid = curNumVertex + j * numNewVerticesEachSide;
				indices[3 * faceid] = vOldId[j];
				indices[3 * faceid + 1] = startVid + 2;  //order of vertex matters! use the same with v1-v2-v3
				indices[3 * faceid + 2] = startVid;
				faceid++;
				indices[3 * faceid] = startVid;
				indices[3 * faceid + 1] = startVid + 2;
				indices[3 * faceid + 2] = startVid + 1;
			}
			else if (jj == RESOLU){
				int startVid = curNumVertex + j * numNewVerticesEachSide + 1 + (jj - 1) * 2;
				indices[3 * faceid] = startVid;
				indices[3 * faceid + 1] = startVid + 1;  //order of vertex matters! use the same with v1-v2-v3
				indices[3 * faceid + 2] = vOldId[(j + 1) % 3];
				faceid++;
				indices[3 * faceid] = startVid;
				indices[3 * faceid + 1] = vOldId[(j + 1) % 3];
				indices[3 * faceid + 2] = curNumVertex + ((j + 1) % 3) * numNewVerticesEachSide;
			}
			else{
				int startVid = curNumVertex + j * numNewVerticesEachSide + 1 + (jj - 1) * 2;
				indices[3 * faceid] = startVid;
				indices[3 * faceid + 1] = startVid + 1;  //order of vertex matters! use the same with v1-v2-v3
				indices[3 * faceid + 2] = startVid + 3;
				faceid++; 
				indices[3 * faceid] = startVid;
				indices[3 * faceid + 1] = startVid + 3;
				indices[3 * faceid + 2] = startVid + 2;
			}
		}
	}

	int startFaceId = curNumFaces + 3 * (RESOLU + 1) * 2;
	for (int j = 0; j < 3; j++){
		if (neighborIds[j] < 0){
			continue;
		}

		for (int jj = 0; jj <= RESOLU; jj++){
			if (jj == 0){
				indices[3 * (startFaceId + jj)] = vOldId[j];
				indices[3 * (startFaceId + jj) + 1] = neighborIds[j];
				indices[3 * (startFaceId + jj) + 2] = curNumVertex + j * numNewVerticesEachSide + 2;
			}
			else if (jj == RESOLU)
			{
				indices[3 * (startFaceId + jj)] = curNumVertex + j * numNewVerticesEachSide + 2 * jj;
				indices[3 * (startFaceId + jj) + 1] = neighborIds[j];
				indices[3 * (startFaceId + jj) + 2] = vOldId[(j + 1) % 3];
			}
			else{
				indices[3 * (startFaceId + jj)] = curNumVertex + j * numNewVerticesEachSide + 2 * jj;
				indices[3 * (startFaceId + jj) + 1] = neighborIds[j];
				indices[3 * (startFaceId + jj) + 2] = curNumVertex + j * numNewVerticesEachSide + 2 * (jj + 1);
			}
		}
		startFaceId += (RESOLU + 1);
	}

	//erase current triangle
	indices[3 * i] = 0;
	indices[3 * i + 1] = 0;
	indices[3 * i + 2] = 0;
}


__global__ void d_disturbVertex_CuboidModel(float* vertexCoords, int vertexcount,
	float3 start, float3 end, float deformationScaleVertical, float3 dir2nd)
	//if a vertex is too close to the cutting plane, then disturb it a little to avoid numerical error
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= vertexcount)	return;

	float3 pos = make_float3(vertexCoords[3 * i], vertexCoords[3 * i + 1], vertexCoords[3 * i + 2]);

	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);

	float thr = 0.000001;
	float disturb = 0.00001;

	float3 n = normalize(cross(dir2nd, tunnelVec));

	float3 voxelVec = pos - start;
	float l = dot(voxelVec, tunnelVec);
	if (l > 0 && l < tunnelLength){
		float l2 = dot(voxelVec, dir2nd);
		if (abs(l2) < deformationScaleVertical){
			float3 prjPoint = start + l*tunnelVec + l2*dir2nd;
			float dis = length(pos - prjPoint);

			//when dis==0 , disturb the vertex a little to avoid numerical error
			if (dis < thr){
				float3 disturbVec = n*disturb;

				vertexCoords[3 * i] = pos.x + disturbVec.x;
				vertexCoords[3 * i + 1] = pos.y + disturbVec.y;
				vertexCoords[3 * i + 2] = pos.z + disturbVec.z;
			}
		}
	}

	return;
}

__global__ void d_modifyMeshKernel_CuboidModel(float* vertexCoords, unsigned int* indices, int facecount, int vertexcount, float* norms, float3 start, float3 end, float r, float deformationScale, float deformationScaleVertical, float3 dir2nd, int* numAddedFaces, float* vertexColorVals)
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

	if (shapeModel == CUBOID){
		int threadsPerBlock = 64;
		int blocksPerGrid = (poly->vertexcount + threadsPerBlock - 1) / threadsPerBlock;
		d_disturbVertex_CuboidModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, poly->vertexcount,
			tunnelStart, tunnelEnd, deformationScaleVertical, rectVerticalDir);
	}
	else if (shapeModel == CIRCLE){
		int threadsPerBlock = 64;
		int blocksPerGrid = (poly->facecount + threadsPerBlock - 1) / threadsPerBlock;
		d_disturbVertex_CircleModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, d_indices, poly->facecountOri, tunnelStart, tunnelEnd);
	}


	int numAddedFaces;
	int numAddedVertices;
	if (shapeModel == CUBOID){
		int threadsPerBlock = 64;
		int blocksPerGrid = (poly->facecount + threadsPerBlock - 1) / threadsPerBlock; 
		d_modifyMeshKernel_CuboidModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, d_indices, poly->facecountOri, poly->vertexcountOri, d_norms,
			tunnelStart, tunnelEnd, deformationScale, deformationScale, deformationScaleVertical, rectVerticalDir,
			d_numAddedFaces, d_vertexColorVals);

		cudaMemcpy(&numAddedFaces, d_numAddedFaces, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "added new face count " << numAddedFaces << std::endl;
		numAddedVertices = numAddedFaces / 3 * 4;
	}
	else if (shapeModel == CIRCLE){
		int threadsPerBlock = 64;
		int blocksPerGrid = (poly->facecount + threadsPerBlock - 1) / threadsPerBlock;

		int *d_numIntersectTris;
		cudaMalloc(&d_numIntersectTris, sizeof(int));
		cudaMemset(d_numIntersectTris, 0, sizeof(int));
		d_modifyMeshKernel_CircledModel_round1 << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, d_indices, poly->facecountOri, tunnelStart, tunnelEnd, d_numIntersectTris, d_intersectedTris);

		int numIntersectTris;
		cudaMemcpy(&numIntersectTris, d_numIntersectTris, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "num of Intersect Triangles: " << numIntersectTris << std::endl;
		if (numIntersectTris > MAX_CIRCLE_INTERACT){
			std::cout << "too many Intersect Triangles "<< std::endl;
			exit(0);
		}

		d_modifyMeshKernel_CircledModel_round2 << <blocksPerGrid, threadsPerBlock >> >(d_indices, poly->facecountOri, tunnelStart, tunnelEnd, numIntersectTris, d_intersectedTris, d_neighborIdsOfIntersectedTris);


		std::vector<uint> l_intersectedTris(MAX_CIRCLE_INTERACT, -1);
		cudaMemcpy(&(l_intersectedTris[0]), d_intersectedTris, sizeof(uint)* MAX_CIRCLE_INTERACT, cudaMemcpyDeviceToHost);

		std::vector<int> l_neighborIdsOfIntersectedTris(MAX_CIRCLE_INTERACT * 3, -1);
		cudaMemcpy(&(l_neighborIdsOfIntersectedTris[0]), d_neighborIdsOfIntersectedTris, sizeof(int)* 3 * MAX_CIRCLE_INTERACT, cudaMemcpyDeviceToHost);

		


		int *d_numAddedVertices;
		cudaMalloc(&d_numAddedVertices, sizeof(int));
		cudaMemset(d_numAddedVertices, 0, sizeof(int));
		threadsPerBlock = 16;
		blocksPerGrid = (numIntersectTris + threadsPerBlock - 1) / threadsPerBlock;
		d_modifyMeshKernel_CircledModel_round3 << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, d_indices, poly->facecountOri, poly->vertexcountOri, d_norms, tunnelStart, tunnelEnd, d_numAddedFaces, d_numAddedVertices, d_vertexColorVals, d_intersectedTris, d_neighborIdsOfIntersectedTris, numIntersectTris);
		cudaMemcpy(&numAddedFaces, d_numAddedFaces, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&numAddedVertices, d_numAddedVertices, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "added new face count " << numAddedFaces << std::endl;
		std::cout << "added new vertice count " << numAddedVertices << std::endl;





		std::vector<float3> vv(numAddedVertices, make_float3(0,0,0));
		cudaMemcpy(&(vv[0].x), d_vertexCoords + poly->vertexcountOri * 3, sizeof(float)* numAddedVertices*3, cudaMemcpyDeviceToHost);

		std::vector<uint3> fff(numAddedFaces, make_uint3(0,0,0));
		cudaMemcpy(&(fff[0].x), d_indices + poly->facecountOri * 3, sizeof(uint)* numAddedFaces * 3, cudaMemcpyDeviceToHost);


		cudaFree(d_numAddedVertices);

		cudaFree(d_numIntersectTris);


	}

	poly->facecount += numAddedFaces;
	poly->vertexcount += numAddedVertices;


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



//////////////////////deform by data type and model////////////////////////////

//////////////////////deform poly

__global__ void d_deformPolyMesh_CuboidModel(float* vertexCoords_init, float* vertexCoords, int vertexcount,
	float3 start, float3 end, float r, float deformationScale, float deformationScaleVertical, float3 dir2nd,
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

__global__ void d_deformPolyMesh_CircleModel(float* vertexCoords_init, float* vertexCoords, int vertexcount,
	float3 start, float3 end, float r, float radius, float* vertexDeviateVals)
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
		float3 prjPoint = start + l*tunnelVec;
		float dis = length(pos - prjPoint);
		//!!NOTE!! the case dis==0 is not processed!! suppose this case will not happen by some spacial preprocessing
		if (dis > 0 && dis < radius){
			float3 dir = normalize(pos - prjPoint);

			float newDis = radius - (radius - dis) / radius * (radius - r);
			float3 newPos = prjPoint + newDis * dir;
			vertexCoords[3 * i] = newPos.x;
			vertexCoords[3 * i + 1] = newPos.y;
			vertexCoords[3 * i + 2] = newPos.z;

			vertexDeviateVals[i] = length(newPos - pos) / (radius / 2); //value range [0,1]
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

	if (shapeModel == CUBOID){
		d_deformPolyMesh_CuboidModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords_init, d_vertexCoords, poly->vertexcount, tunnelStart, tunnelEnd, degree, deformationScale, deformationScaleVertical, rectVerticalDir, d_vertexDeviateVals);

	}
	else if (shapeModel == CIRCLE){
		d_deformPolyMesh_CircleModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords_init, d_vertexCoords, poly->vertexcount, tunnelStart, tunnelEnd, degree, radius, d_vertexDeviateVals);
	}

	cudaMemcpy(poly->vertexCoords, d_vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);
	if (isColoringDeformedPart)
	{
		cudaMemcpy(poly->vertexDeviateVals, d_vertexDeviateVals, sizeof(float)*poly->vertexcount, cudaMemcpyDeviceToHost);
	}
}

//////////////////////deform particle

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

//////////////////////deform volume
__global__ void
d_deformVolume_CircleModel(cudaExtent volumeSize, float3 start, float3 end, float3 spacing, float r, float radius){
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
		float dis = sqrt(disToStart*disToStart - l*l);

		if (dis < r){
			float res = 0;
			surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
		}
		else if (dis < radius){
			float3 prjPoint = start + l*tunnelVec;
			float3 dir = normalize(pos - prjPoint);
			float3 samplePos = prjPoint + dir* (dis - r) / (radius - r)*radius;
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
	return;
}

__global__ void
d_deformVolume_CuboidModel(cudaExtent volumeSize, float3 start, float3 end, float3 spacing, float r, float deformationScale, float deformationScaleVertical, float3 dir2nd)
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
	if (shapeModel == CUBOID){
		d_deformVolume_CuboidModel << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, degree, deformationScale, deformationScaleVertical, rectVerticalDir);
	}
	else if (shapeModel == CIRCLE){
		d_deformVolume_CircleModel << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, r, radius);
	}
	checkCudaErrors(cudaUnbindTexture(volumeTexInput));
}

void PositionBasedDeformProcessor::doVolumeDeform2Tunnel(float degreeOpen, float degreeClose)
{
	cudaExtent size = volume->volumeCuda.size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp(size.width, blockSize.x), iDivUp(size.height, blockSize.y), iDivUp(size.depth, blockSize.z));

	cudaChannelFormatDesc cd = volume->volumeCudaOri.channelDesc;

	checkCudaErrors(cudaBindTextureToArray(volumeTexInput, volume->volumeCudaOri.content, cd));
	checkCudaErrors(cudaBindSurfaceToArray(volumeSurfaceOut, volumeCudaIntermediate->content));
	if (shapeModel == CUBOID){
		d_deformVolume_CuboidModel << <gridSize, blockSize >> >(size, lastTunnelStart, lastTunnelEnd, volume->spacing, degreeClose, deformationScale, deformationScaleVertical, lastDeformationDirVertical);
	}
	else if (shapeModel == CIRCLE){
		d_deformVolume_CircleModel << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, degreeClose, radius);
	}

	checkCudaErrors(cudaUnbindTexture(volumeTexInput));

	checkCudaErrors(cudaBindTextureToArray(volumeTexInput, volumeCudaIntermediate->content, cd));
	checkCudaErrors(cudaBindSurfaceToArray(volumeSurfaceOut, volume->volumeCuda.content));
	if (shapeModel == CUBOID){
		d_deformVolume_CuboidModel << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, degreeOpen, deformationScale, deformationScaleVertical, rectVerticalDir);
	}
	else if (shapeModel == CIRCLE){
		d_deformVolume_CircleModel << <gridSize, blockSize >> >(size, tunnelStart, tunnelEnd, volume->spacing, degreeOpen, radius);
	}

	checkCudaErrors(cudaUnbindTexture(volumeTexInput));
}



//////////////////////deform selector////////////////////////////

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
		std::cout << " deformDataByDegree not implemented " << std::endl;
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
		std::cout << " deformDataByDegree2Tunnel not implemented " << std::endl;
		exit(0);
	}

	return;
}








//isForceDeform is not supported yet
bool PositionBasedDeformProcessor::process(float* modelview, float* projection, int winWidth, int winHeight)
{
	if (!isActive)
		return false;

	float3 eyeInLocal = matrixMgr->getEyeInLocal();

	if (isForceDeform){
		if (systemState == ORIGINAL){
			systemState = OPENING;
			computeTunnelInfo(eyeInLocal);
			tunnelTimer1.init(outTime, 0);

			if (dataType == MESH){ //for poly data, the original data will be modified, which is not applicable to other types of data
				modifyPolyMesh();
			}
		}
		else if (systemState == OPENING){
			if (tunnelTimer1.out()){
				systemState = DEFORMED;
				tunnelTimer1.end();
				r = deformationScale / 2; //reset r
			}
		}
		else if (systemState == DEFORMED){
		}
		else{
			std::cout << "STATE NOT DEFINED for force deform" << std::endl;
			exit(0);
		}
	}
	else{
		if (systemState == ORIGINAL){
			if (!atProperLocation(eyeInLocal, true)){
				systemState = OPENING;
				computeTunnelInfo(eyeInLocal);
				tunnelTimer1.init(outTime, 0);

				if (dataType == MESH){ //for poly data, the original data will be modified, which is not applicable to other types of data
					modifyPolyMesh();
				}

			}
		}
		else if (systemState == OPENING){
			if (tunnelTimer1.out()){
				systemState = DEFORMED;
				tunnelTimer1.end();
				r = deformationScale / 2; //reset r
			}
			else{
				if (atProperLocation(eyeInLocal, true)){
					systemState = CLOSING;
					float passed = tunnelTimer1.getTime();
					tunnelTimer1.init(outTime, (passed >= outTime) ? 0 : (outTime - passed));
				}
				else if (!atProperLocation(eyeInLocal, false)){
					systemState = MIXING;
					float passed = tunnelTimer1.getTime();
					tunnelTimer2.init(outTime, (passed >= outTime) ? 0 : (outTime - passed));
					tunnelTimer1.init(outTime, 0);
				}
			}
		}
		else if (systemState == CLOSING){
			if (tunnelTimer1.out()){
				systemState = ORIGINAL;
				tunnelTimer1.end();
				r = 0; //reset r
				resetData();
			}
			else if (!atProperLocation(eyeInLocal, true)){
				storeCurrentTunnel();
				computeTunnelInfo(eyeInLocal);
				if (sameTunnel()){
					systemState = OPENING;
					float passed = tunnelTimer1.getTime();
					tunnelTimer1.init(outTime, outTime - passed);
				}
				else{
					systemState = MIXING;
					//tunnelTimer2 = tunnelTimer1;//cannot assign in this way!!!
					tunnelTimer2.init(outTime, tunnelTimer1.getTime());
					tunnelTimer1.init(outTime, 0);
				}
			}
		}
		else if (systemState == MIXING){
			if (tunnelTimer1.out() && tunnelTimer2.out()){
				systemState = DEFORMED;
				tunnelTimer1.end();
				tunnelTimer2.end();
				r = deformationScale / 2; //reset r
			}
			else if (atProperLocation(eyeInLocal, true)){
				//tunnelTimer2 may not have been out() yet, but here ignore the 2nd tunnel
				systemState = CLOSING;
				float passed = tunnelTimer1.getTime();
				tunnelTimer1.init(outTime, (passed >= outTime) ? 0 : (outTime - passed));
			}
			else{
				//new mixture or old mixture. todo
			}
		}
		else if (systemState == DEFORMED){
			if (atProperLocation(eyeInLocal, true)){
				systemState = CLOSING;
				float passed = tunnelTimer1.getTime();
				tunnelTimer1.init(outTime, (passed >= outTime) ? 0 : (outTime - passed));
			}
			else if (!atProperLocation(eyeInLocal, false)){
				storeCurrentTunnel();
				computeTunnelInfo(eyeInLocal);
				systemState = MIXING;
				tunnelTimer2.init(outTime, 0);
				tunnelTimer1.init(outTime, 0);
			}
		}
		else{
			std::cout << "STATE NOT DEFINED" << std::endl;
			exit(0);
		}
	}

	if (systemState == MIXING){
		if (tunnelTimer2.out()){
			r = tunnelTimer1.getTime() / outTime * deformationScale / 2;
			deformDataByDegree(r);
			//std::cout << "doing mixinig with r: " << r << " and 0" << std::endl;
		}
		else{
			if (tunnelTimer1.out()){
				std::cout << "impossible combination!" << std::endl;
				exit(0);
			}
			else{
				float rOpen = tunnelTimer1.getTime() / outTime * deformationScale / 2;
				float rClose = (1 - tunnelTimer2.getTime() / outTime) * deformationScale / 2;
				deformDataByDegree2Tunnel(rOpen, rClose);
				//std::cout << "doing mixinig with r: " << rOpen << " and " << rClose << std::endl;
			}
		}
	}
	else if (systemState == OPENING){
		r = tunnelTimer1.getTime() / outTime * deformationScale / 2;
		deformDataByDegree(r);
		//std::cout << "doing openning with r: " << r << std::endl;
	}
	else if (systemState == CLOSING){
		r = (1 - tunnelTimer1.getTime() / outTime) * deformationScale / 2;
		deformDataByDegree(r);
		//std::cout << "doing closing with r: " << r << std::endl;
	}

	return false;
}






