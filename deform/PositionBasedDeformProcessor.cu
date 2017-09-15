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

};

void PositionBasedDeformProcessor::PrepareDataStructureForPolyDeform()
{
	if (!d_vertexCoords) { cudaMalloc(&d_vertexCoords, sizeof(float)*poly->vertexcount * 3 * 2); };
	if (!d_vertexCoords_init){ cudaMalloc(&d_vertexCoords_init, sizeof(float)*poly->vertexcount * 3 * 2); };
	if (!d_indices){ cudaMalloc(&d_indices, sizeof(unsigned int)*poly->facecount * 3 * 2); };
	if (!d_indices_init){ cudaMalloc(&d_indices_init, sizeof(unsigned int)*poly->facecount * 3 * 2); };
	if (!d_norms){ cudaMalloc(&d_norms, sizeof(float)*poly->vertexcount * 3 * 2); };
	if (!d_vertexDeviateVals){ cudaMalloc(&d_vertexDeviateVals, sizeof(float)*poly->vertexcount * 2); };
	if (!d_vertexColorVals) { cudaMalloc(&d_vertexColorVals, sizeof(float)*poly->vertexcount * 2); };
	if (!d_numAddedFaces){ cudaMalloc(&d_numAddedFaces, sizeof(int)); };

	////NOTE!! here doubled the space. Hopefully it is large enough
	cudaMemcpy(d_vertexCoords, poly->vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vertexCoords_init, poly->vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, poly->indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices_init, poly->indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_norms, poly->vertexNorms, sizeof(float)*poly->vertexcount * 3, cudaMemcpyHostToDevice);
	cudaMemset(d_vertexDeviateVals, 0, sizeof(float)*poly->vertexcount * 2);
	cudaMemcpy(d_vertexColorVals, poly->vertexColorVals, sizeof(float)*poly->vertexcount, cudaMemcpyHostToDevice);
	cudaMemset(d_numAddedFaces, 0, sizeof(int));

	prepareIntersectionInfoForCircleAndPoly();
}

void PositionBasedDeformProcessor::polyMeshDataUpdated()
{
	PrepareDataStructureForPolyDeform();
	if (systemState != ORIGINAL && isActive){
		modifyPolyMesh();
		doPolyDeform(r);
	}

	//!!!!!  NOT COMPLETE YET!!! to add more state changes!!!!
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
	
	d_vec_mid.resize(particle->numParticles);

	if (particle->orientation.size() == 0){
		std::cout << "processing unoriented particles" << std::endl;
	}
	d_vec_orientation.assign(&(particle->orientation[0]), &(particle->orientation[0]) + particle->numParticles);
	d_vec_lastFramePos.assign(&(particle->pos[0]), &(particle->pos[0]) + particle->numParticles);

}

void PositionBasedDeformProcessor::particleDataUpdated()
{
	d_vec_posOrig.assign(&(particle->pos[0]), &(particle->pos[0]) + particle->numParticles);
	if (d_vec_mid.size() != particle->numParticles){
		d_vec_mid.resize(particle->numParticles);
	}

	if (particle->orientation.size() >0){
		d_vec_orientation.assign(&(particle->orientation[0]), &(particle->orientation[0]) + particle->numParticles);
	}
	else{
		std::cout << "WARNING! particles orientation expected but not found" << std::endl;
	}

	if (systemState != ORIGINAL && isActive){
		std::cout << "camera BAD in new original data" << std::endl;


		if (!atProperLocation(matrixMgr->getEyeInLocal(), true)){
			computeTunnelInfo(matrixMgr->getEyeInLocal());
		}
		else{
			adjustTunnelEnds();
		}
		
		if (systemState == MIXING)
		{
			adjustTunnelEndsLastTunnel();
		}

		if (systemState == MIXING){
			doParticleDeform2Tunnel(rOpen, rClose);
		}
		else{
			doParticleDeform(r); //although the deformation might be computed later, still do it once here to compute d_vec_posTarget in case of needed by detecting proper location
		}
	}
	else{
		d_vec_posTarget.assign(&(particle->pos[0]), &(particle->pos[0]) + particle->numParticles);
		std::cout << "camera GOOD in new original data" << std::endl;
	}
}

void PositionBasedDeformProcessor::resetData()
{
	if (dataType == VOLUME){
		volume->reset();
	}
	else if (dataType == MESH){
		poly->reset();
		PrepareDataStructureForPolyDeform(); //must do it once now, since 
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
	float step;
	if (dataType == VOLUME){
		step = 1; //for VOLUME, no need to be less than 1
	}
	else if (dataType == PARTICLE){
		step = 0.5; //for VOLUME, no need to be less than 1
	}
	else{
		step = 0.25;
	}

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

		tunnelEnd = centerPoint + tunnelAxis*step;
		while (!atProperLocation(tunnelEnd, true)){
			tunnelEnd += tunnelAxis*step;
		}
		tunnelStart = centerPoint;
		while (!atProperLocation(tunnelStart, true)){
			tunnelStart -= tunnelAxis*step;
		}


		/* //new method

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

void PositionBasedDeformProcessor::adjustTunnelEnds()
{
	float step;
	if (dataType == VOLUME){
		step = 1; //for VOLUME, no need to be less than 1
	}
	else if (dataType == PARTICLE){
		step = 0.5; //for VOLUME, no need to be less than 1
	}
	else{
		step = 0.25;
	}

	if (isForceDeform) //just for testing, may not be precise
	{
		return;
	}
	else{
		float3 tunnelAxis = normalize(tunnelEnd - tunnelStart);

		tunnelEnd -= tunnelAxis*step; //originally should be improper
		if (atProperLocation(tunnelEnd, true)){//should shorten
			while (atProperLocation(tunnelEnd, true) && dot(tunnelEnd - tunnelStart, tunnelAxis)>0){
				tunnelEnd -= tunnelAxis*step;
			}
			tunnelEnd += tunnelAxis*step; //one step backwards
		}
		else{
			while (!atProperLocation(tunnelEnd, true)){
				tunnelEnd += tunnelAxis*step;
			}
		}

		tunnelStart += tunnelAxis*step;//originally should be improper
		if (atProperLocation(tunnelStart, true)){//should shorten
			while (atProperLocation(tunnelStart, true) && dot(tunnelEnd - tunnelStart, tunnelAxis)>0){
				tunnelStart += tunnelAxis*step;
			}
			tunnelStart -= tunnelAxis*step;
		}
		else{
			while (!atProperLocation(tunnelStart, true)){
				tunnelStart -= tunnelAxis*step;
			}
		}
	}
}

void PositionBasedDeformProcessor::adjustTunnelEndsLastTunnel()
{
	float step;
	if (dataType == VOLUME){
		step = 1; //for VOLUME, no need to be less than 1
	}
	else if (dataType == PARTICLE){
		step = 0.5; //for VOLUME, no need to be less than 1
	}
	else{
		step = 0.25;
	}

	if (isForceDeform) //just for testing, may not be precise
	{
		return;
	}
	else{
		lastTunnelStart, lastTunnelEnd;
		float3 tunnelAxis = normalize(lastTunnelEnd - lastTunnelStart);

		lastTunnelEnd -= tunnelAxis*step; //originally should be improper
		if (atProperLocation(lastTunnelEnd, true)){//should shorten
			while (atProperLocation(lastTunnelEnd, true) && dot(lastTunnelEnd - lastTunnelStart, tunnelAxis)>0){
				lastTunnelEnd -= tunnelAxis*step;
			}
			lastTunnelEnd += tunnelAxis*step; //one step backwards
		}
		else{
			while (!atProperLocation(lastTunnelEnd, true)){
				lastTunnelEnd += tunnelAxis*step;
			}
		}

		lastTunnelStart += tunnelAxis*step;//originally should be improper
		if (atProperLocation(lastTunnelStart, true)){//should shorten
			while (atProperLocation(lastTunnelStart, true) && dot(lastTunnelEnd - lastTunnelStart, tunnelAxis)>0){
				lastTunnelStart += tunnelAxis*step;
			}
			lastTunnelStart -= tunnelAxis*step;
		}
		else{
			while (!atProperLocation(lastTunnelStart, true)){
				lastTunnelStart -= tunnelAxis*step;
			}
		}
	}
}


bool PositionBasedDeformProcessor::sameTunnel(){
	float thr = 0.00001;
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


struct functor_ParticleDis
{
	float3 pos;
	float thrAlong, thrPerpen;
	template<typename Tuple>
	__device__ __host__ void operator() (Tuple t){
		float3 center = make_float3(thrust::get<0>(t));
		float3 orient = thrust::get<1>(t);

		float l = dot(pos - center, orient);
		float3 proj = center + l*orient;
		float l3 = length(proj - pos);

		if (abs(l) < thrAlong && l3 < thrPerpen){
			thrust::get<2>(t) = 1;
		}
		else{
			thrust::get<2>(t) = 0;
		}
		
	}
	
	functor_ParticleDis(float3 _pos, float _thrAlong, float _thrPerpen)
		: pos(_pos), thrAlong(_thrAlong), thrPerpen(_thrPerpen){}
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

__global__ void d_checkIfTooCloseToPoly(float3 pos, uint* indices, int faceCoords, float* vertexCoords, float *norms, float thr, bool* res)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= faceCoords)	return;

	uint3 inds = make_uint3(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);
	float3 v1 = make_float3(vertexCoords[3 * inds.x], vertexCoords[3 * inds.x + 1], vertexCoords[3 * inds.x + 2]);
	float3 v2 = make_float3(vertexCoords[3 * inds.y], vertexCoords[3 * inds.y + 1], vertexCoords[3 * inds.y + 2]);
	float3 v3 = make_float3(vertexCoords[3 * inds.z], vertexCoords[3 * inds.z + 1], vertexCoords[3 * inds.z + 2]);

	//float dis = min(min(length(pos - v1), length(pos - v2)), length(pos - v3));
	float dis = d_disToTri(pos, v1, v2, v3, thr);

	float3 norm1 = make_float3(norms[3 * inds.x], norms[3 * inds.x + 1], norms[3 * inds.x + 2]);
	float3 norm2 = make_float3(norms[3 * inds.y], norms[3 * inds.y + 1], norms[3 * inds.y + 2]);
	float3 norm3 = make_float3(norms[3 * inds.z], norms[3 * inds.z + 1], norms[3 * inds.z + 2]);
	float3 avenorm = (norm1 + norm2 + norm3) / 3;
	//if (dot(avenorm, pos - (v1 + v2 + v3) / 3) < 0){//back side of the triangle
	//	if (dis < thr/2)
	//	{
	//		*res = true;
	//	}
	//}
	//else{
		if (dis < thr)
		{
			*res = true;
		}
	//}
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
			d_checkIfTooCloseToPoly << <blocksPerGrid, threadsPerBlock >> >(pos, d_indices, poly->facecount, d_vertexCoords_init, d_norms, disThr, d_tooCloseToData);
		}
		else{
			d_checkIfTooCloseToPoly << <blocksPerGrid, threadsPerBlock >> >(pos, d_indices, poly->facecount, d_vertexCoords, d_norms, disThr, d_tooCloseToData);
		}

		bool tooCloseToData;
		cudaMemcpy(&tooCloseToData, d_tooCloseToData, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		cudaFree(d_tooCloseToData);
		return !tooCloseToData;
	}
	else if (dataType == PARTICLE){
		float init = 10000;
		float inSavePosition;
		//if (useOriData){
		//	inSavePosition = thrust::transform_reduce(
		//		d_vec_posOrig.begin(), d_vec_posOrig.end(), functor_dis(pos), init, thrust::minimum<float>());
		//}
		//else{
		//	inSavePosition = thrust::transform_reduce(
		//		d_vec_posTarget.begin(), d_vec_posTarget.end(), functor_dis(pos), init, thrust::minimum<float>());
		//}
		//return (inSavePosition > disThr);
		
		if (useOriData){
			thrust::for_each(
				thrust::make_zip_iterator(
				thrust::make_tuple(
				d_vec_posOrig.begin(),
				d_vec_orientation.begin(),
				d_vec_mid.begin()
				)),
				thrust::make_zip_iterator(
				thrust::make_tuple(
				d_vec_posOrig.end(),
				d_vec_orientation.end(),
				d_vec_mid.end()
				)),
				functor_ParticleDis(pos, thrOriented[0], thrOriented[1]));
		}
		else{
			thrust::for_each(
				thrust::make_zip_iterator(
				thrust::make_tuple(
				d_vec_posTarget.begin(),
				d_vec_orientation.begin(),
				d_vec_mid.begin()
				)),
				thrust::make_zip_iterator(
				thrust::make_tuple(
				d_vec_posTarget.end(),
				d_vec_orientation.end(),
				d_vec_mid.end()
				)),
				functor_ParticleDis(pos, thrOriented[0], thrOriented[1]));
		}

		float result = thrust::reduce(thrust::device,
			d_vec_mid.begin(), d_vec_mid.end(),
			-1,
			thrust::maximum<float>());

		return (result < 0.5);

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
	float disturb = 0.000002;

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


__global__ void d_disturbVertices_CircleModel(float* vertexCoords, unsigned int* indices, int vertexcount, float3 start, float3 end)
//if a vertex or an egde is too close to the cutting axis, then disturb it a little to avoid numerical error
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= vertexcount)	return;

	float3 pos = make_float3(vertexCoords[3 * i], vertexCoords[3 * i + 1], vertexCoords[3 * i + 2]);

	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);

	float thr = 0.0001;
	float disturb = 0.0002;

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

			vertexCoords[3 * i] = pos.x;
			vertexCoords[3 * i + 1] = pos.y;
			vertexCoords[3 * i + 2] = pos.z;
		}
	}
	return;

}


__global__ void d_disturbEdges_CircleModel(float* vertexCoords, unsigned int* indices, int facecount, float3 start, float3 end, float *d_tt)
//if a vertex or an egde is too close to the cutting axis, then disturb it a little to avoid numerical error
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= facecount)	return;
	uint3 inds = make_uint3(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);
	float3 v1 = make_float3(vertexCoords[3 * inds.x], vertexCoords[3 * inds.x + 1], vertexCoords[3 * inds.x + 2]);
	float3 v2 = make_float3(vertexCoords[3 * inds.y], vertexCoords[3 * inds.y + 1], vertexCoords[3 * inds.y + 2]);
	float3 v3 = make_float3(vertexCoords[3 * inds.z], vertexCoords[3 * inds.z + 1], vertexCoords[3 * inds.z + 2]);

	float thr = 0.0001;
	float disturb = 0.0002;
	float3 tunnelVec = normalize(end - start);

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
		d_tt[i] = 2;
		return;
	}

	float3 intersect = start + dis_startToIntersectionPoint * tunnelVec;
	bool isInside = false;
	float numericalThr = 0.000001;
	//if (dot(cross(l12, -l31), cross(l12, intersect - v1)) >= 0){
	//	if (dot(cross(l23, -l12), cross(l23, intersect - v2)) >= 0){
	//		if (dot(cross(l31, -l23), cross(l31, intersect - v3)) >= 0){
	if (dot(cross(l12, -l31), cross(l12, intersect - v1)) >= -numericalThr){
		if (dot(cross(l23, -l12), cross(l23, intersect - v2)) >= -numericalThr){
			if (dot(cross(l31, -l23), cross(l31, intersect - v3)) >= -numericalThr){
				isInside = true;
			}
		}
	}
	if (!isInside){
		d_tt[i] = 1;
		return;
	}

	d_tt[i] = 3;
	{
		float3 prjOn12 = v1 + dot(intersect - v1, l12) * l12;
		float disTo12 = length(intersect - prjOn12);
		if (disTo12 < thr){ //<= numericalThr){//== 0){
			float3 refVec = make_float3(0, 0, 1);
			//make sure the refVec and the edge are not perpendicular
			if (abs(dot(refVec, l12)) < 0.01){
				refVec = make_float3(0, 1, 0);
			}
			if (abs(dot(refVec, l12)) < 0.01){
				refVec = make_float3(1, 0, 0);
			}

			//make sure the edge is pointing to the same direction as the refVec
			float3 vEgde = l12;
			if (dot(refVec, vEgde) < 0){
				vEgde = -vEgde;
			}
			float3 disturbVec = normalize(cross(vEgde, tunnelVec))*disturb;
			
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
		float3 prjOn23 = v2 + dot(intersect - v2, l23) * l23;
		float disTo23 = length(intersect - prjOn23);
		if (disTo23 < thr){ // <= numericalThr){// == 0){
			float3 refVec = make_float3(0, 0, 1);
			//make sure the refVec and the edge are not perpendicular
			if (abs(dot(refVec, l23)) < 0.01){
				refVec = make_float3(0, 1, 0);
			}
			if (abs(dot(refVec, l23)) < 0.01){
				refVec = make_float3(1, 0, 0);
			}

			//make sure the edge is pointing to the same direction as the refVec
			float3 vEgde = l23;
			if (dot(refVec, vEgde) < 0){
				vEgde = -vEgde;
			}
			float3 disturbVec = normalize(cross(vEgde, tunnelVec))*disturb;

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
		float3 prjOn31 = v3 + dot(intersect - v3, l31) * l31;
		float disTo31 = length(intersect - prjOn31);
		if (disTo31 < thr){ //<= numericalThr){//== 0){
			float3 refVec = make_float3(0, 0, 1);
			//make sure the refVec and the edge are not perpendicular
			if (abs(dot(refVec, l31)) < 0.01){
				refVec = make_float3(0, 1, 0);
			}
			if (abs(dot(refVec, l31)) < 0.01){
				refVec = make_float3(1, 0, 0);
			}

			//make sure the edge is pointing to the same direction as the refVec
			float3 vEgde = l31;
			if (dot(refVec, vEgde) < 0){
				vEgde = -vEgde;
			}
			float3 disturbVec = normalize(cross(vEgde, tunnelVec))*disturb;

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



	/*{
		float3 prjOn12 = v1 + dot(intersect - v1, l12) * l12;
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
		float3 prjOn23 = v2 + dot(intersect - v2, l23) * l23;
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
		float3 prjOn31 = v3 + dot(intersect - v3, l31) * l31;
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
	}*/

	return;
}

__global__ void d_disturb_CircleModel(float* vertexCoords, unsigned int* indices, int facecount, float3 start, float3 end)
//if a vertex or an egde is too close to the cutting axis, then disturb it a little to avoid numerical error
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= facecount)	return;
	uint3 inds = make_uint3(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);
	float3 v1 = make_float3(vertexCoords[3 * inds.x], vertexCoords[3 * inds.x + 1], vertexCoords[3 * inds.x + 2]);
	float3 v2 = make_float3(vertexCoords[3 * inds.y], vertexCoords[3 * inds.y + 1], vertexCoords[3 * inds.y + 2]);
	float3 v3 = make_float3(vertexCoords[3 * inds.z], vertexCoords[3 * inds.z + 1], vertexCoords[3 * inds.z + 2]);
	
	float thr = 0.00001;
	float disturb = 0.00002;
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


__device__ float3 d_intersectPoint(float3 o, float3 d, float3 a, float3 b) //get the intersected point of ray from point o to direction d, and segment a-b. if parallel, return (-1000, -1000, -1000)
//assume d is normalized
{
	//https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/

	float3 v1 = o - a;
	float3 v2 = b - a;
	if (abs(dot(normalize(v2), d)) > 0.99999999){
		return make_float3(-1000, -1000, -1000);
	}

	float3 v3 =  - normalize(cross(d, cross(d, v2)));
	float t1 = length(cross(v2, v1)) / abs(dot(v2, v3));
	return o + d*t1;
}

__global__ void d_modifyMeshKernel_CircledModel_round3(float* vertexCoords, unsigned int* indices, int facecount, int vertexcount, float* norms, float3 start, float3 end, int* numAddedFaces, int* numAddedVertices, float* vertexColorVals, unsigned int* intersectedTris, int* neighborIdsOfIntersectedTris, int numIntersectTris, int* minThr)
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


	float thr = 0.00001;
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
	
	triNormal = -triNormal; //bad design... should fix it at the first time it is used

	float3 nNewToOld[3] = { normalize(v1 - intersect), normalize(v2 - intersect), normalize(v3 - intersect) };
	float angles[3] = { acosf(dot(nNewToOld[0], nNewToOld[1])), acosf(dot(nNewToOld[1], nNewToOld[2])), acosf(dot(nNewToOld[2], nNewToOld[0])) };
	float rotateMat[3][9];

	float halfRotateAngles[3] = { angles[0] / (RESOLU + 1.0) / 2.0, angles[1] / (RESOLU + 1.0) / 2.0, angles[2] / (RESOLU + 1.0) / 2.0 };
	float minArcRatio = cosf(min(min(halfRotateAngles[0], halfRotateAngles[1]), halfRotateAngles[2]));
	int minArcRatioInt = minArcRatio * 1000000;
	atomicMin(minThr, minArcRatioInt);

	for (int j = 0; j < 3; j++){
		for (int jj = 1; jj <= RESOLU; jj++){
			float rotateAngles[3] = { angles[0] * jj / (RESOLU + 1.0), angles[1] * jj / (RESOLU + 1.0), angles[2] * jj / (RESOLU + 1.0) };
			float cosRotateAngles[3] = { cosf(rotateAngles[0]), cosf(rotateAngles[1]), cosf(rotateAngles[2]) };
			float sinRotateAngles[3] = { sinf(rotateAngles[0]), sinf(rotateAngles[1]), sinf(rotateAngles[2]) };

			//http://scipp.ucsc.edu/~haber/ph216/rotation_12.pdf
			//for (int j = 0; j < 3; j++){
				rotateMat[j][0] = cosRotateAngles[j] + triNormal.x*triNormal.x*(1 - cosRotateAngles[j]);
				rotateMat[j][1] = triNormal.x*triNormal.y*(1 - cosRotateAngles[j]) - triNormal.z*sinRotateAngles[j];
				rotateMat[j][2] = triNormal.x*triNormal.z*(1 - cosRotateAngles[j]) + triNormal.y*sinRotateAngles[j];
				rotateMat[j][3] = triNormal.x*triNormal.y*(1 - cosRotateAngles[j]) + triNormal.z*sinRotateAngles[j];
				rotateMat[j][4] = cosRotateAngles[j] + triNormal.y*triNormal.y*(1 - cosRotateAngles[j]);
				rotateMat[j][5] = triNormal.y*triNormal.z*(1 - cosRotateAngles[j]) - triNormal.x*sinRotateAngles[j];
				rotateMat[j][6] = triNormal.x*triNormal.z*(1 - cosRotateAngles[j]) - triNormal.y*sinRotateAngles[j];
				rotateMat[j][7] = triNormal.y*triNormal.z*(1 - cosRotateAngles[j]) + triNormal.x*sinRotateAngles[j];
				rotateMat[j][8] = cosRotateAngles[j] + triNormal.z*triNormal.z*(1 - cosRotateAngles[j]);
			//}

			//float3 temp1 = vNew[j] * (1 - ratio1) + vNew[(j + 1) % 3] * ratio1;
			//rotate first
			//nNewToOld[j] = mul(rotateMat[j], nNewToOld[j]);
			//float3 temp1 = intersect + nNewToOld[j] * thr;
			float3 v = mul(rotateMat[j], nNewToOld[j]);
			float3 temp1 = intersect + v * thr;

			int curId = curNumVertex + j * numNewVerticesEachSide + 1 + (jj-1) * 2;
			
			vertexCoords[3 * curId] = temp1.x;
			vertexCoords[3 * curId + 1] = temp1.y;
			vertexCoords[3 * curId + 2] = temp1.z;
			norms[3 * curId] = newNormal.x;
			norms[3 * curId + 1] = newNormal.y;
			norms[3 * curId + 2] = newNormal.z;
			vertexColorVals[curId] = newColorVal;

			curId++;
			//float ratio1 = 1.0 * jj / (RESOLU + 1.0);
			//temp1 = vOld[j] * (1 - ratio1) + vOld[(j + 1) % 3] * ratio1;
			temp1 = d_intersectPoint(intersect, v, vOld[j], vOld[(j + 1) % 3]);
			float ratio1 = length(temp1 - vOld[j]) / length(vOld[(j + 1) % 3] - vOld[j]);
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

__global__ void d_modifyMeshKernel_CircledModel_round3_new(float* vertexCoords, unsigned int* indices, int facecount, int vertexcount, float* norms, float3 start, float3 end, int* numAddedFaces, int* numAddedVertices, float* vertexColorVals, unsigned int* intersectedTris, int* neighborIdsOfIntersectedTris, int numIntersectTris, int* minThr)
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

	int numNewVerticesEachSide = 1 + RESOLU;//1 for small triangle; RESOLU on the arc; RESOLU on the original edge
	int numAddedVerticesBefore = atomicAdd(numAddedVertices, numNewVerticesEachSide * 3);
	int curNumVertex = vertexcount + numAddedVerticesBefore;

	float disto1 = length(intersect - v1);
	float disto2 = length(intersect - v2);
	float disto3 = length(intersect - v3);
	float disAllDouble = 2 * (disto1 + disto2 + disto3);
	float3 newNormal = (disto2 + disto3) / disAllDouble*norm1 + (disto1 + disto3) / disAllDouble*norm2 + (disto2 + disto1) / disAllDouble*norm3; //simple interpolation
	float newColorVal = vertexColorVals[inds.x];//any one of the 3 vertices. assume the 3 values are the same for a triangle


	float thr = 0.00001;
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

	triNormal = -triNormal; //bad design... should fix it at the first time it is used

	float3 nNewToOld[3] = { normalize(v1 - intersect), normalize(v2 - intersect), normalize(v3 - intersect) };
	float angles[3] = { acosf(dot(nNewToOld[0], nNewToOld[1])), acosf(dot(nNewToOld[1], nNewToOld[2])), acosf(dot(nNewToOld[2], nNewToOld[0])) };
	float rotateMat[3][9];

	float halfRotateAngles[3] = { angles[0] / (RESOLU + 1.0) / 2.0, angles[1] / (RESOLU + 1.0) / 2.0, angles[2] / (RESOLU + 1.0) / 2.0 };
	float minArcRatio = cosf(min(min(halfRotateAngles[0], halfRotateAngles[1]), halfRotateAngles[2]));
	int minArcRatioInt = minArcRatio * 1000000;
	atomicMin(minThr, minArcRatioInt);

	for (int j = 0; j < 3; j++){
		for (int jj = 1; jj <= RESOLU; jj++){
			float rotateAngles[3] = { angles[0] * jj / (RESOLU + 1.0), angles[1] * jj / (RESOLU + 1.0), angles[2] * jj / (RESOLU + 1.0) };
			float cosRotateAngles[3] = { cosf(rotateAngles[0]), cosf(rotateAngles[1]), cosf(rotateAngles[2]) };
			float sinRotateAngles[3] = { sinf(rotateAngles[0]), sinf(rotateAngles[1]), sinf(rotateAngles[2]) };

			//http://scipp.ucsc.edu/~haber/ph216/rotation_12.pdf
			//for (int j = 0; j < 3; j++){
			rotateMat[j][0] = cosRotateAngles[j] + triNormal.x*triNormal.x*(1 - cosRotateAngles[j]);
			rotateMat[j][1] = triNormal.x*triNormal.y*(1 - cosRotateAngles[j]) - triNormal.z*sinRotateAngles[j];
			rotateMat[j][2] = triNormal.x*triNormal.z*(1 - cosRotateAngles[j]) + triNormal.y*sinRotateAngles[j];
			rotateMat[j][3] = triNormal.x*triNormal.y*(1 - cosRotateAngles[j]) + triNormal.z*sinRotateAngles[j];
			rotateMat[j][4] = cosRotateAngles[j] + triNormal.y*triNormal.y*(1 - cosRotateAngles[j]);
			rotateMat[j][5] = triNormal.y*triNormal.z*(1 - cosRotateAngles[j]) - triNormal.x*sinRotateAngles[j];
			rotateMat[j][6] = triNormal.x*triNormal.z*(1 - cosRotateAngles[j]) - triNormal.y*sinRotateAngles[j];
			rotateMat[j][7] = triNormal.y*triNormal.z*(1 - cosRotateAngles[j]) + triNormal.x*sinRotateAngles[j];
			rotateMat[j][8] = cosRotateAngles[j] + triNormal.z*triNormal.z*(1 - cosRotateAngles[j]);
			//}

			//float3 temp1 = vNew[j] * (1 - ratio1) + vNew[(j + 1) % 3] * ratio1;
			//rotate first
			//nNewToOld[j] = mul(rotateMat[j], nNewToOld[j]);
			//float3 temp1 = intersect + nNewToOld[j] * thr;
			float3 v = mul(rotateMat[j], nNewToOld[j]);
			float3 temp1 = intersect + v * thr;

			int curId = curNumVertex + j * numNewVerticesEachSide + 1 + (jj - 1);

			vertexCoords[3 * curId] = temp1.x;
			vertexCoords[3 * curId + 1] = temp1.y;
			vertexCoords[3 * curId + 2] = temp1.z;
			norms[3 * curId] = newNormal.x;
			norms[3 * curId + 1] = newNormal.y;
			norms[3 * curId + 2] = newNormal.z;
			vertexColorVals[curId] = newColorVal;
		}
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

	float thr = 0.00001;
	float disturb = 0.00002;

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
	//cudaMemcpy(d_vertexCoords, poly->vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_indices, poly->indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_norms, poly->vertexNorms, sizeof(float)*poly->vertexcount * 3, cudaMemcpyHostToDevice);

	//cudaMemset(d_vertexDeviateVals, 0, sizeof(float)*poly->vertexcount * 2);
	//cudaMemcpy(d_vertexColorVals, poly->vertexColorVals, sizeof(float)*poly->vertexcount, cudaMemcpyHostToDevice);

	//cudaMemset(d_numAddedFaces, 0, sizeof(int));

	if (shapeModel == CUBOID){
		int threadsPerBlock = 64;
		int blocksPerGrid = (poly->vertexcount + threadsPerBlock - 1) / threadsPerBlock;
		d_disturbVertex_CuboidModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, poly->vertexcount,
			tunnelStart, tunnelEnd, deformationScaleVertical, rectVerticalDir);
	}
	else if (shapeModel == CIRCLE){
		//int threadsPerBlock = 64;
		//int blocksPerGrid = (poly->facecount + threadsPerBlock - 1) / threadsPerBlock;
		//d_disturb_CircleModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, d_indices, poly->facecountOri, tunnelStart, tunnelEnd);
		int threadsPerBlock = 64;
		int blocksPerGrid = (poly->vertexcount + threadsPerBlock - 1) / threadsPerBlock;
		d_disturbVertices_CircleModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, d_indices, poly->vertexcount, tunnelStart, tunnelEnd);


		float *d_tt;
		cudaMalloc(&d_tt, sizeof(float)* poly->facecountOri);
		cudaMemset(d_tt, 0, sizeof(float)*poly->facecountOri);



		threadsPerBlock = 64;
		blocksPerGrid = (poly->facecount + threadsPerBlock - 1) / threadsPerBlock;
		d_disturbEdges_CircleModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, d_indices, poly->facecount, tunnelStart, tunnelEnd, d_tt);



		std::vector<float> xxx(poly->facecountOri, 0);
		cudaMemcpy(&(xxx[0]), d_tt, sizeof(float)* poly->facecountOri, cudaMemcpyDeviceToHost);
		int y = 9;
		y++;


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
			std::cout << "too many Intersect Triangles " << std::endl;
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

		int *d_minThr; //using int because cuda atomicMin() does not suppport float?
		cudaMalloc(&d_minThr, sizeof(int));
		int temp = 10000000;
		cudaMemcpy(d_minThr, &temp, sizeof(int), cudaMemcpyHostToDevice);


		threadsPerBlock = 16;
		blocksPerGrid = (numIntersectTris + threadsPerBlock - 1) / threadsPerBlock;
		d_modifyMeshKernel_CircledModel_round3 << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, d_indices, poly->facecountOri, poly->vertexcountOri, d_norms, tunnelStart, tunnelEnd, d_numAddedFaces, d_numAddedVertices, d_vertexColorVals, d_intersectedTris, d_neighborIdsOfIntersectedTris, numIntersectTris, d_minThr);
		cudaMemcpy(&numAddedFaces, d_numAddedFaces, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&numAddedVertices, d_numAddedVertices, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "added new face count " << numAddedFaces << std::endl;
		std::cout << "added new vertice count " << numAddedVertices << std::endl;

		cudaMemcpy(&temp, d_minThr, sizeof(int), cudaMemcpyDeviceToHost);

		circleThr = temp * 1.0 / 1000000 * 0.95; //0.95 is a selected parameter
		std::cout << "circleThr: " << circleThr << std::endl;


		if (numAddedVertices > 0){
			std::vector<float3> vv(numAddedVertices, make_float3(0, 0, 0));
			cudaMemcpy(&(vv[0].x), d_vertexCoords + poly->vertexcountOri * 3, sizeof(float)* numAddedVertices * 3, cudaMemcpyDeviceToHost);

			std::vector<uint3> fff(numAddedFaces, make_uint3(0, 0, 0));
			cudaMemcpy(&(fff[0].x), d_indices + poly->facecountOri * 3, sizeof(uint)* numAddedFaces * 3, cudaMemcpyDeviceToHost);
		}

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

	int d1 = poly->facecountOri, d2 = poly->facecountOri + RESOLU + 1 + RESOLU, d3 = poly->facecountOri + 2 * (RESOLU + 1 + RESOLU);
	intersect = (
		make_float3(poly->vertexCoords[3 * d1], poly->vertexCoords[3 * d1 + 1], poly->vertexCoords[3 * d1 + 2]) +
		make_float3(poly->vertexCoords[3 * d2], poly->vertexCoords[3 * d2 + 1], poly->vertexCoords[3 * d2 + 2]) +
		make_float3(poly->vertexCoords[3 * d3], poly->vertexCoords[3 * d3 + 1], poly->vertexCoords[3 * d3 + 2])) / 3;
	usedFaceCount = poly->facecount;
	usedVertexCount = poly->vertexcount;
	cudaMemcpy(d_indices_init, d_indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyDeviceToDevice);
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




//http://geomalgorithms.com/a07-_distance.html#dist3D_Segment_to_Segment
__device__ float4 dist3D_Segment_to_Segment(float3 S1_0, float3 S1_1, float3 S2_0, float3 S2_1) //return (closest point on S2, plus the closest distance)
{
	float SMALL_NUM = 0.00000001;
	float3   u = S1_1 - S1_0;
	float3   v = S2_1 - S2_0;
	float3   w = S1_0 - S2_0;
	float    a = dot(u, u);         // always >= 0
	float    b = dot(u, v);
	float    c = dot(v, v);         // always >= 0
	float    d = dot(u, w);
	float    e = dot(v, w);
	float    D = a*c - b*b;        // always >= 0
	float    sc, sN, sD = D;       // sc = sN / sD, default sD = D >= 0
	float    tc, tN, tD = D;       // tc = tN / tD, default tD = D >= 0

	// compute the line parameters of the two closest points
	if (D < SMALL_NUM) { // the lines are almost parallel
		sN = 0.0;         // force using point P0 on segment S1
		sD = 1.0;         // to prevent possible division by 0.0 later
		tN = e;
		tD = c;
	}
	else {                 // get the closest points on the infinite lines
		sN = (b*e - c*d);
		tN = (a*e - b*d);
		if (sN < 0.0) {        // sc < 0 => the s=0 edge is visible
			sN = 0.0;
			tN = e;
			tD = c;
		}
		else if (sN > sD) {  // sc > 1  => the s=1 edge is visible
			sN = sD;
			tN = e + b;
			tD = c;
		}
	}

	if (tN < 0.0) {            // tc < 0 => the t=0 edge is visible
		tN = 0.0;
		// recompute sc for this edge
		if (-d < 0.0)
			sN = 0.0;
		else if (-d > a)
			sN = sD;
		else {
			sN = -d;
			sD = a;
		}
	}
	else if (tN > tD) {      // tc > 1  => the t=1 edge is visible
		tN = tD;
		// recompute sc for this edge
		if ((-d + b) < 0.0)
			sN = 0;
		else if ((-d + b) > a)
			sN = sD;
		else {
			sN = (-d + b);
			sD = a;
		}
	}
	// finally do the division to get sc and tc
	sc = (abs(sN) < SMALL_NUM ? 0.0 : sN / sD);
	tc = (abs(tN) < SMALL_NUM ? 0.0 : tN / tD);

	// get the difference of the two closest points
	float3   dP = w + (sc * u) - (tc * v);  // =  S1(sc) - S2(tc)

	//return length(dP);   // return the closest distance
	return make_float4(S2_0 + (tc * v), length(dP));
}
//===================================================================


__global__ void d_checkEdge_afterDeformPolyMeshByCircleModel(float* vertexCoords, unsigned int * indices, int facecount, 	float3 start, float3 end, float r, float circleThr, int4* errorEdgeInfo, float3* closestPoint) //errorEdgeInfo:(faceid, edgevertex1, edgevertex2)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= facecount)	return;

	uint3 inds = make_uint3(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);
	float3 v1 = make_float3(vertexCoords[3 * inds.x], vertexCoords[3 * inds.x + 1], vertexCoords[3 * inds.x + 2]);
	float3 v2 = make_float3(vertexCoords[3 * inds.y], vertexCoords[3 * inds.y + 1], vertexCoords[3 * inds.y + 2]);
	float3 v3 = make_float3(vertexCoords[3 * inds.z], vertexCoords[3 * inds.z + 1], vertexCoords[3 * inds.z + 2]);

	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);

	{
		float3 sToP1 = v1 - start, sToP2 = v2 - start;
		float l1 = dot(tunnelVec, sToP1), l2 = dot(tunnelVec, sToP2);
		if ((l1 <= -tunnelLength && l2 <= -tunnelLength) || (l1 >= tunnelLength && l2 >= tunnelLength))
		{
		}
		else{
			float4 disvec= dist3D_Segment_to_Segment(start, end, v1, v2);
			float dis = disvec.w;
			if (dis < circleThr * r){
				*errorEdgeInfo = make_int4(i, inds.x, inds.y, inds.z);
				*closestPoint = make_float3(disvec);
				return;
			}
		}			
	}

	{
		float3 sToP1 = v2 - start, sToP2 = v3 - start;
		float l1 = dot(tunnelVec, sToP1), l2 = dot(tunnelVec, sToP2);
		if ((l1 <= -tunnelLength && l2 <= -tunnelLength) || (l1 >= tunnelLength && l2 >= tunnelLength))
		{
		}
		else{
			float4 disvec = dist3D_Segment_to_Segment(start, end, v2, v3);
			float dis = disvec.w;	
			if (dis < circleThr * r){
				*errorEdgeInfo = make_int4(i, inds.y, inds.z, inds.x);
				*closestPoint = make_float3(disvec);
				return;
			}
		}
	}

	{
		float3 sToP1 = v3 - start, sToP2 = v1 - start;
		float l1 = dot(tunnelVec, sToP1), l2 = dot(tunnelVec, sToP2);
		if ((l1 <= -tunnelLength && l2 <= -tunnelLength) || (l1 >= tunnelLength && l2 >= tunnelLength))
		{
		}
		else{
			float4 disvec = dist3D_Segment_to_Segment(start, end, v3, v1);
			float dis = disvec.w;		
			if (dis < circleThr * r){
				*errorEdgeInfo = make_int4(i, inds.z, inds.x, inds.y);
				*closestPoint = make_float3(disvec);
				return;
			}
		}
	}
	return;
}



__global__ void d_checkEdge_afterDeformPolyMeshByCircleModel_new(float* vertexCoords, unsigned int * indices, int facecount, float3 start, float3 end, float r, float circleThr, int* numErrorEdges, int4* errorEdgeInfo, float3* closestPoint) //errorEdgeInfo:(faceid, edgevertex1, edgevertex2)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= facecount)	return;

	uint3 inds = make_uint3(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);
	float3 v1 = make_float3(vertexCoords[3 * inds.x], vertexCoords[3 * inds.x + 1], vertexCoords[3 * inds.x + 2]);
	float3 v2 = make_float3(vertexCoords[3 * inds.y], vertexCoords[3 * inds.y + 1], vertexCoords[3 * inds.y + 2]);
	float3 v3 = make_float3(vertexCoords[3 * inds.z], vertexCoords[3 * inds.z + 1], vertexCoords[3 * inds.z + 2]);

	float3 tunnelVec = normalize(end - start);
	float tunnelLength = length(end - start);

	{
		float3 sToP1 = v1 - start, sToP2 = v2 - start;
		float l1 = dot(tunnelVec, sToP1), l2 = dot(tunnelVec, sToP2);
		if ((l1 <= -tunnelLength && l2 <= -tunnelLength) || (l1 >= tunnelLength && l2 >= tunnelLength))
		{
		}
		else{
			float4 disvec = dist3D_Segment_to_Segment(start, end, v1, v2);
			float dis = disvec.w;
			if (dis < circleThr * r){
				int id = atomicAdd(numErrorEdges, 1);
				if (id < MAX_ERROR_EDGE){
					*(errorEdgeInfo+id) = make_int4(i, inds.x, inds.y, inds.z);
					*(closestPoint + id) = make_float3(disvec);
				}
				return;
			}
		}
	}

	{
		float3 sToP1 = v2 - start, sToP2 = v3 - start;
		float l1 = dot(tunnelVec, sToP1), l2 = dot(tunnelVec, sToP2);
		if ((l1 <= -tunnelLength && l2 <= -tunnelLength) || (l1 >= tunnelLength && l2 >= tunnelLength))
		{
		}
		else{
			float4 disvec = dist3D_Segment_to_Segment(start, end, v2, v3);
			float dis = disvec.w;
			if (dis < circleThr * r){
				int id = atomicAdd(numErrorEdges, 1);
				if (id < MAX_ERROR_EDGE){
					*(errorEdgeInfo + id) = make_int4(i, inds.y, inds.z, inds.x);
					*(closestPoint + id) = make_float3(disvec);
				}
				//*errorEdgeInfo = make_int4(i, inds.x, inds.y, inds.z);
				//*closestPoint = make_float3(disvec); 
				return;
			}
		}
	}

	{
		float3 sToP1 = v3 - start, sToP2 = v1 - start;
		float l1 = dot(tunnelVec, sToP1), l2 = dot(tunnelVec, sToP2);
		if ((l1 <= -tunnelLength && l2 <= -tunnelLength) || (l1 >= tunnelLength && l2 >= tunnelLength))
		{
		}
		else{
			float4 disvec = dist3D_Segment_to_Segment(start, end, v3, v1);
			float dis = disvec.w;
			if (dis < circleThr * r){
				int id = atomicAdd(numErrorEdges, 1);
				if (id < MAX_ERROR_EDGE){
					*(errorEdgeInfo + id) = make_int4(i, inds.z, inds.x, inds.y);
					*(closestPoint + id) = make_float3(disvec);
				}
				/**errorEdgeInfo = make_int4(i, inds.x, inds.y, inds.z);
				*closestPoint = make_float3(disvec); */
				return;
			}
		}
	}
	return;
}

__global__ void removeDup(int* numErrorEdges, int4* errorEdgeInfo, float3* closestPoints, int* d_anotherFaceOfErrorEdge)
{
	int realCount = 0;
	for (int i = 0; i < *numErrorEdges; i++){
		bool noDup = true;
		for (int j = 0; j < realCount && noDup; j++){
			if ((errorEdgeInfo[j].y == errorEdgeInfo[i].y &&errorEdgeInfo[j].z == errorEdgeInfo[i].z)
				|| (errorEdgeInfo[j].z == errorEdgeInfo[i].y &&errorEdgeInfo[j].y == errorEdgeInfo[i].z)){
				noDup = false;
				d_anotherFaceOfErrorEdge[j] = errorEdgeInfo[i].x;
			}
		}
		if (noDup){			
			if (i > realCount){
				errorEdgeInfo[realCount] = errorEdgeInfo[i];
				closestPoints[realCount] = closestPoints[i];
			}
			realCount++;
		}
	}
	*numErrorEdges = realCount;
}

__global__ void d_reworkErrorEdges_new(float* vertexCoords, int vertexcount, unsigned int* indices, int facecount, int4* errorEdgeInfos, int* d_face2, float* norms, float* vertexColorVals, float3* d_closestPoint, float3 intersect, float r, float radius, int numErrorEdges, int* addedVertices, int * addedFaces)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numErrorEdges)	return;

	int4 errorEdgeInfo = errorEdgeInfos[i];
	float3 closestPoint = d_closestPoint[i];

	const int N = 5;
	
	//we know the order of the vertices in the face of errorEdgeInfo.x, but not in face2
	int inde1 = errorEdgeInfo.y, inde2 = errorEdgeInfo.z;
	int indtop1 = errorEdgeInfo.w;
	//float3 v1 = make_float3(vertexCoords[3 * indtop1], vertexCoords[3 * indtop1 + 1], vertexCoords[3 * indtop1 + 2]);
	float3 ve1 = make_float3(vertexCoords[3 * inde1], vertexCoords[3 * inde1 + 1], vertexCoords[3 * inde1 + 2]);
	float3 ve2 = make_float3(vertexCoords[3 * inde2], vertexCoords[3 * inde2 + 1], vertexCoords[3 * inde2 + 2]);

	float chord = sqrt(r*r - length(closestPoint - intersect) *length(closestPoint - intersect) ) * 2;
	float seg = length(closestPoint - ve1) - chord / 2;
	float3 chordDir = normalize(ve2 - ve1);
	float3 left = ve1 + chordDir * seg;
	//float3 right = ve2 - normalize(ve2 - ve1) * seg;

	
	int addedVerticesBefore = atomicAdd(addedVertices, N);
	int vid = vertexcount + addedVerticesBefore;
	for (int i = 0; i < N; i++){
		float3 curp = left + chordDir* chord / (N + 1) * (i + 1);
		float3 newv, newNormal;
		float newColVal;
		float targetDis = r* 1.001;
		float3 outvec = normalize(curp - intersect);
		newv = intersect + outvec *targetDis;

		float ratio = length(curp - ve1) / (length(curp - ve1) + length(curp - ve2));
		newNormal = make_float3(norms[3 * inde1], norms[3 * inde1 + 1], norms[3 * inde1 + 2]) * (1 - ratio) + make_float3(norms[3 * inde2], norms[3 * inde2 + 1], norms[3 * inde2 + 2]) * ratio;
		newColVal = vertexColorVals[inde1] * (1 - ratio) + vertexColorVals[inde2] * ratio;

		norms[3 * (vid + i)] = newNormal.x;
		norms[3 * (vid + i) + 1] = newNormal.y;
		norms[3 * (vid + i) + 2] = newNormal.z;
		vertexColorVals[(vid + i)] = newColVal;
		vertexCoords[3 * (vid + i)] = newv.x;
		vertexCoords[3 * (vid + i) + 1] = newv.y;
		vertexCoords[3 * (vid + i) + 2] = newv.z;
	}

	
	int addedFacesBefore = atomicAdd(addedFaces, N + 1);
	int fid = facecount + addedFacesBefore;
	for (int i = 0; i <= N; i++){
		if (i == 0){
			indices[3 * (fid+i)] = inde1;
			indices[3 * (fid + i) + 1] = vid+i;
			indices[3 * (fid + i) + 2] = indtop1;
		}
		else if (i == N){
			indices[3 * (fid + i)] = vid + i - 1;
			indices[3 * (fid + i) + 1] = inde2;
			indices[3 * (fid + i) + 2] = indtop1;
		}
		else{
			indices[3 * (fid + i)] = vid + i - 1;
			indices[3 * (fid + i) + 1] = vid + i;
			indices[3 * (fid + i) + 2] = indtop1;
		}
	}

	//erase old face
	indices[3 * errorEdgeInfo.x] = 0;
	indices[3 * errorEdgeInfo.x + 1] = 0;
	indices[3 * errorEdgeInfo.x + 2] = 0;

	int faceid2 = d_face2[i];
	if (faceid2 > -1){
		uint3 inds = make_uint3(indices[3 * faceid2], indices[3 * faceid2 + 1], indices[3 * faceid2 + 2]);
		int inde1, inde2, indtop1;
		if ((inds.x == errorEdgeInfo.y && inds.y == errorEdgeInfo.z) || (inds.y == errorEdgeInfo.y && inds.x == errorEdgeInfo.z)){
			inde1 = inds.x, inde2 = inds.y, indtop1 = inds.z;
		}
		else if ((inds.x == errorEdgeInfo.y && inds.z == errorEdgeInfo.z) || (inds.z == errorEdgeInfo.y && inds.x == errorEdgeInfo.z)){
			inde1 = inds.z, inde2 = inds.x, indtop1 = inds.y;
		}
		else if ((inds.y == errorEdgeInfo.y && inds.z == errorEdgeInfo.z) || (inds.z == errorEdgeInfo.y && inds.y == errorEdgeInfo.z)){
			inde1 = inds.y, inde2 = inds.z, indtop1 = inds.x;
		}

		int addedFacesBefore = atomicAdd(addedFaces, N+1);
		int fid = facecount + addedFacesBefore;
		for (int i = 0; i < N; i++){
			if (i == 0){
				indices[3 * (fid + i)] = inde1;
				indices[3 * (fid + i) + 1] = vid + i;
				indices[3 * (fid + i) + 2] = indtop1;
			}
			else if (i == N){
				indices[3 * (fid + i)] = vid + i - 1;
				indices[3 * (fid + i) + 1] = inde2;
				indices[3 * (fid + i) + 2] = indtop1;
			}
			else{
				indices[3 * (fid + i)] = vid + i - 1;
				indices[3 * (fid + i) + 1] = vid + i;
				indices[3 * (fid + i) + 2] = indtop1;
			}
		}

		//erase old face
		indices[3 * errorEdgeInfo.x] = 0;
		indices[3 * errorEdgeInfo.x + 1] = 0;
		indices[3 * errorEdgeInfo.x + 2] = 0;

		//erase old face
		indices[3 * faceid2] = 0;
		indices[3 * faceid2 + 1] = 0;
		indices[3 * faceid2 + 2] = 0;
	}

	return;
}



__global__ void d_findAnotherFaceOfErrorEdge(unsigned int* indices, int facecount, int4 errorEdgeInfo, int* face2)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= facecount)	return;
	if (i == errorEdgeInfo.x) return;

	uint3 inds = make_uint3(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);

	if ((inds.x == errorEdgeInfo.y && inds.y == errorEdgeInfo.z) || (inds.y == errorEdgeInfo.y && inds.x == errorEdgeInfo.z)){
		*face2 = i;
	}
	else if ((inds.x == errorEdgeInfo.y && inds.z == errorEdgeInfo.z) || (inds.z == errorEdgeInfo.y && inds.x == errorEdgeInfo.z)){
		*face2 = i;
	}
	else if ((inds.y == errorEdgeInfo.y && inds.z == errorEdgeInfo.z) || (inds.z == errorEdgeInfo.y && inds.y == errorEdgeInfo.z)){
		*face2 = i;
	}
	return;		
}


__global__ void d_reworkErrorEdges(float* vertexCoords, float* vertexCoords_init, int vertexcount, unsigned int* indices, int facecount, int4 errorEdgeInfo, int* face2, float* norms, float* vertexColorVals, float3* d_closestPoint, float3 intersect, float r, float radius)
{
	//we know the order of the vertices in the face of errorEdgeInfo.x, but not in face2
	int inde1 = errorEdgeInfo.y, inde2 = errorEdgeInfo.z;
	int indtop1 = errorEdgeInfo.w;

	//float3 v1 = make_float3(vertexCoords[3 * indtop1], vertexCoords[3 * indtop1 + 1], vertexCoords[3 * indtop1 + 2]);
	float3 ve1 = make_float3(vertexCoords[3 * inde1], vertexCoords[3 * inde1 + 1], vertexCoords[3 * inde1 + 2]);
	float3 ve2 = make_float3(vertexCoords[3 * inde2], vertexCoords[3 * inde2 + 1], vertexCoords[3 * inde2 + 2]);
	
	//float3 newv = (make_float3(vertexCoords[3 * inde1], vertexCoords[3 * inde1 + 1], vertexCoords[3 * inde1 + 2]) + make_float3(vertexCoords[3 * inde2], vertexCoords[3 * inde2 + 1], vertexCoords[3 * inde2 + 2])) / 2;
	//float3 newNormal = (make_float3(norms[3 * inde1], norms[3 * inde1 + 1], norms[3 * inde1 + 2]) + make_float3(norms[3 * inde2], norms[3 * inde2 + 1], norms[3 * inde2 + 2])) / 2;
	//float newColVal = (vertexColorVals[inde1] + vertexColorVals[inde2]) / 2;

	float3 closestPoint = *d_closestPoint;
	float3 newv, newv_init, newNormal;
	float newColVal;

	if (length(closestPoint - ve1) < 0.00001 || length(closestPoint - ve2) < 0.00001){
		newv = (make_float3(vertexCoords[3 * inde1], vertexCoords[3 * inde1 + 1], vertexCoords[3 * inde1 + 2]) + make_float3(vertexCoords[3 * inde2], vertexCoords[3 * inde2 + 1], vertexCoords[3 * inde2 + 2])) / 2;
		newv_init = (make_float3(vertexCoords_init[3 * inde1], vertexCoords_init[3 * inde1 + 1], vertexCoords_init[3 * inde1 + 2]) + make_float3(vertexCoords_init[3 * inde2], vertexCoords_init[3 * inde2 + 1], vertexCoords_init[3 * inde2 + 2])) / 2;
		newNormal = (make_float3(norms[3 * inde1], norms[3 * inde1 + 1], norms[3 * inde1 + 2]) + make_float3(norms[3 * inde2], norms[3 * inde2 + 1], norms[3 * inde2 + 2])) / 2;
		newColVal = (vertexColorVals[inde1] + vertexColorVals[inde2]) / 2;
	}
	else{
		//newv = closestPoint;
		float targetDis = r* 1.001;
		float3 outvec = normalize(closestPoint - intersect);
			newv = intersect + outvec *targetDis;

		float ratio = length(closestPoint - ve1) / (length(closestPoint - ve1) + length(closestPoint - ve2));
		newNormal = make_float3(norms[3 * inde1], norms[3 * inde1 + 1], norms[3 * inde1 + 2]) * (1-ratio) + make_float3(norms[3 * inde2], norms[3 * inde2 + 1], norms[3 * inde2 + 2]) * ratio;
		newColVal = vertexColorVals[inde1] * (1 - ratio) + vertexColorVals[inde2] * ratio;
		//newv_init = make_float3(vertexCoords_init[3 * inde1], vertexCoords_init[3 * inde1 + 1], vertexCoords_init[3 * inde1 + 2]) * (1 - ratio) + make_float3(vertexCoords_init[3 * inde2], vertexCoords_init[3 * inde2 + 1], vertexCoords_init[3 * inde2 + 2]) * ratio;
		newv_init = intersect + outvec* (targetDis - r) / (radius - r)*radius;

	}

	int newId = vertexcount;
	norms[3 * newId] = newNormal.x;
	norms[3 * newId + 1] = newNormal.y;
	norms[3 * newId + 2] = newNormal.z;
	vertexColorVals[newId] = newColVal;
	vertexCoords[3 * newId] = newv.x;
	vertexCoords[3 * newId + 1] = newv.y;
	vertexCoords[3 * newId + 2] = newv.z;
	vertexCoords_init[3 * newId] = newv_init.x;
	vertexCoords_init[3 * newId + 1] = newv_init.y;
	vertexCoords_init[3 * newId + 2] = newv_init.z;


	int faceid = facecount;
	indices[3 * faceid] = inde1;
	indices[3 * faceid + 1] = newId ;
	indices[3 * faceid + 2] = indtop1;
	faceid++;
	indices[3 * faceid] = newId;
	indices[3 * faceid + 1] = inde2;
	indices[3 * faceid + 2] = indtop1;

	//erase old face
	indices[3 * errorEdgeInfo.x] = 0;
	indices[3 * errorEdgeInfo.x + 1] = 0;
	indices[3 * errorEdgeInfo.x + 2] = 0;

	if (*face2 > -1){
		int faceid2 = *face2;
		uint3 inds = make_uint3(indices[3 * faceid2], indices[3 * faceid2 + 1], indices[3 * faceid2 + 2]);
		int inde1, inde2, indtop1;
		if ((inds.x == errorEdgeInfo.y && inds.y == errorEdgeInfo.z) || (inds.y == errorEdgeInfo.y && inds.x == errorEdgeInfo.z)){
			inde1 = inds.x, inde2 = inds.y, indtop1 = inds.z;
		}
		else if ((inds.x == errorEdgeInfo.y && inds.z == errorEdgeInfo.z) || (inds.z == errorEdgeInfo.y && inds.x == errorEdgeInfo.z)){
			inde1 = inds.z, inde2 = inds.x, indtop1 = inds.y;
		}
		else if ((inds.y == errorEdgeInfo.y && inds.z == errorEdgeInfo.z) || (inds.z == errorEdgeInfo.y && inds.y == errorEdgeInfo.z)){
			inde1 = inds.y, inde2 = inds.z, indtop1 = inds.x;
		}

		faceid++;
		indices[3 * faceid] = inde1;
		indices[3 * faceid + 1] = newId;
		indices[3 * faceid + 2] = indtop1;
		faceid++;
		indices[3 * faceid] = newId;
		indices[3 * faceid + 1] = inde2;
		indices[3 * faceid + 2] = indtop1;

		//erase old face
		indices[3 * faceid2] = 0;
		indices[3 * faceid2 + 1] = 0;
		indices[3 * faceid2 + 2] = 0;
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
		int threadsPerBlock = 64;
		int blocksPerGrid = (usedVertexCount + threadsPerBlock - 1) / threadsPerBlock;

		d_deformPolyMesh_CircleModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords_init, d_vertexCoords, usedVertexCount, tunnelStart, tunnelEnd, degree, radius, d_vertexDeviateVals);

		if (degree > 0.1){
			int *d_numErrorEdges; 
			int numErrorEdges;
			cudaMalloc(&d_numErrorEdges, sizeof(int));
			cudaMemset(d_numErrorEdges, 0, sizeof(int));

			threadsPerBlock = 64;
			blocksPerGrid = (usedFaceCount + threadsPerBlock - 1) / threadsPerBlock;

			d_checkEdge_afterDeformPolyMeshByCircleModel_new << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, d_indices_init, usedFaceCount, tunnelStart, tunnelEnd, degree, circleThr, d_numErrorEdges, d_errorEdgeInfos, d_closestPoints); //errorEdgeInfo:(faceid, edgevertex1, edgevertex2)
			cudaMemcpy(&numErrorEdges, d_numErrorEdges, sizeof(int), cudaMemcpyDeviceToHost);


	
			if (numErrorEdges >0){
				std::vector<float3> qq(MAX_ERROR_EDGE, make_float3(-1, -2, -3));
				cudaMemcpy(&(qq[0].x), d_closestPoints, sizeof(float)* 3 * MAX_ERROR_EDGE, cudaMemcpyDeviceToHost);

				std::vector<int4> aa(MAX_ERROR_EDGE, make_int4(-1,-2,-3,-4));
				cudaMemcpy(&(aa[0].x), d_errorEdgeInfos, sizeof(int)* 4 * MAX_ERROR_EDGE, cudaMemcpyDeviceToHost);
				

				//std::cout << "found bad edge at face " << errorEdgeInfo.x << ", with vertices " << errorEdgeInfo.y << " " << errorEdgeInfo.z << std::endl;
				std::cout << "found bad edge count: " << numErrorEdges << std::endl;

				std::vector<int> temp22(MAX_ERROR_EDGE, -1);
				cudaMemcpy(d_anotherFaceOfErrorEdge, &(temp22[0]), sizeof(int)* MAX_ERROR_EDGE, cudaMemcpyHostToDevice);
				removeDup << <1, 1 >> >(d_numErrorEdges, d_errorEdgeInfos, d_closestPoints, d_anotherFaceOfErrorEdge);
				cudaMemcpy(&numErrorEdges, d_numErrorEdges, sizeof(int), cudaMemcpyDeviceToHost); //numErrorEdges will be reduced


			/*	cudaMemcpy(&tempface2, d_face2, sizeof(int), cudaMemcpyDeviceToHost);
				std::cout << "the 2nd bad edge at face " << tempface2 << std::endl;*/


				//std::vector<uint3> ddd(poly->facecount);
				//cudaMemcpy(&(ddd[0].x), d_indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyDeviceToHost);

				threadsPerBlock = 32;
				blocksPerGrid = (numErrorEdges + threadsPerBlock - 1) / threadsPerBlock;


				int *d_addf; 
				cudaMalloc(&d_addf, sizeof(int));
				cudaMemset(d_addf, 0, sizeof(int));
				int *d_addv;
				cudaMalloc(&d_addv, sizeof(int));
				cudaMemset(d_addv, 0, sizeof(int));


				cudaMemcpy(d_indices, d_indices_init, sizeof(unsigned int)*usedFaceCount * 3, cudaMemcpyDeviceToDevice);


				d_reworkErrorEdges_new << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, usedVertexCount, d_indices, usedFaceCount, d_errorEdgeInfos, d_anotherFaceOfErrorEdge, d_norms, d_vertexColorVals, d_closestPoints, intersect, r, radius, numErrorEdges, d_addv, d_addf);
				
				int numAddedVertices;
				int numAddedFaces;
				cudaMemcpy(&numAddedVertices, d_addv, sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(&numAddedFaces, d_addf, sizeof(int), cudaMemcpyDeviceToHost);


				std::cout << "added new face count " << numAddedFaces << std::endl;
				std::cout << "added new vertice count " << numAddedVertices << std::endl;
				poly->vertexcount += numAddedVertices;
				poly->facecount += numAddedFaces;
				std::cout << "current face count " << poly->facecount << std::endl;
				std::cout << "current vertice count " << poly->vertexcount << std::endl;

				std::vector<float3> bbb(numAddedVertices);
				cudaMemcpy(&(bbb[0].x), d_vertexCoords + usedVertexCount * 3, sizeof(float)*numAddedVertices * 3, cudaMemcpyDeviceToHost);
				std::vector<uint3> aaa(numAddedFaces);
				cudaMemcpy(&(aaa[0].x), d_indices + usedFaceCount * 3, sizeof(unsigned int)*numAddedFaces * 3, cudaMemcpyDeviceToHost);

			

			/*	std::vector<uint3> aaa(poly->facecount);
				cudaMemcpy(&(aaa[0].x), d_indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyDeviceToHost);
				std::vector<float3> bbb(poly->vertexcount);
				cudaMemcpy(&(bbb[0].x), d_vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);
				std::vector<float3> ccc(poly->vertexcount);
				cudaMemcpy(&(ccc[0].x), d_vertexCoords_init, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);*/

				cudaMemcpy(poly->indices, d_indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyDeviceToHost);
				cudaMemcpy(poly->vertexNorms, d_norms, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);
				cudaMemcpy(poly->vertexColorVals, d_vertexColorVals, sizeof(float)*poly->vertexcount, cudaMemcpyDeviceToHost);
			}
		}
	}

	cudaMemcpy(poly->vertexCoords, d_vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);

	if (isColoringDeformedPart)
	{
		cudaMemcpy(poly->vertexDeviateVals, d_vertexDeviateVals, sizeof(float)*poly->vertexcount, cudaMemcpyDeviceToHost);
	}
}





void PositionBasedDeformProcessor::doPolyDeform2(float degree)
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
		//d_deformPolyMesh_CircleModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords_init, d_vertexCoords, usedVertexCount, tunnelStart, tunnelEnd, degree, radius, d_vertexDeviateVals);

		if (degree > 0.1){
			int4 *d_errorEdgeInfo;
			cudaMalloc(&d_errorEdgeInfo, sizeof(int4));
			int4 errorEdgeInfo = make_int4(-1, -1, -1, -1);
			cudaMemcpy(d_errorEdgeInfo, &(errorEdgeInfo.x), sizeof(int4), cudaMemcpyHostToDevice);


			float3 *d_closestPoint;
			cudaMalloc(&d_closestPoint, sizeof(float3));


			threadsPerBlock = 64;
			blocksPerGrid = (poly->facecount + threadsPerBlock - 1) / threadsPerBlock;

			d_checkEdge_afterDeformPolyMeshByCircleModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, d_indices, poly->facecount, tunnelStart, tunnelEnd, degree, circleThr, d_errorEdgeInfo, d_closestPoint); //errorEdgeInfo:(faceid, edgevertex1, edgevertex2)

			float3 closestPoint;
			cudaMemcpy(&(closestPoint.x), d_closestPoint, sizeof(float3), cudaMemcpyDeviceToHost);

			cudaMemcpy(&(errorEdgeInfo.x), d_errorEdgeInfo, sizeof(int4), cudaMemcpyDeviceToHost);
			if (errorEdgeInfo.x > -1){
				std::cout << "found bad edge at face " << errorEdgeInfo.x << ", with vertices " << errorEdgeInfo.y << " " << errorEdgeInfo.z << std::endl;

				int* d_face2;
				cudaMalloc(&d_face2, sizeof(int));
				int tempface2 = -1;
				cudaMemcpy(d_face2, &tempface2, sizeof(int), cudaMemcpyHostToDevice);

				d_findAnotherFaceOfErrorEdge << <blocksPerGrid, threadsPerBlock >> >(d_indices, poly->facecount, errorEdgeInfo, d_face2);

				cudaMemcpy(&tempface2, d_face2, sizeof(int), cudaMemcpyDeviceToHost);
				std::cout << "the 2nd bad edge at face " << tempface2 << std::endl;

				std::vector<uint3> ddd(poly->facecount);
				cudaMemcpy(&(ddd[0].x), d_indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyDeviceToHost);


				d_reworkErrorEdges << <	1, 1 >> >(d_vertexCoords, d_vertexCoords_init, poly->vertexcount, d_indices, poly->facecount, errorEdgeInfo, d_face2, d_norms, d_vertexColorVals, d_closestPoint, intersect, r, radius);
				int numAddedVertices = 1;
				int numAddedFaces = ((tempface2 > (-1)) ? 4 : 2);


				std::cout << "added new face count " << numAddedFaces << std::endl;
				std::cout << "added new vertice count " << numAddedVertices << std::endl;
				poly->vertexcount += numAddedVertices;
				poly->facecount += numAddedFaces;
				std::cout << "current face count " << poly->facecount << std::endl;
				std::cout << "current vertice count " << poly->vertexcount << std::endl;

				cudaFree(d_face2);

				std::vector<uint3> aaa(poly->facecount);
				cudaMemcpy(&(aaa[0].x), d_indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyDeviceToHost);
				std::vector<float3> bbb(poly->vertexcount);
				cudaMemcpy(&(bbb[0].x), d_vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);
				std::vector<float3> ccc(poly->vertexcount);
				cudaMemcpy(&(ccc[0].x), d_vertexCoords_init, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);

				cudaMemcpy(poly->indices, d_indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyDeviceToHost);
				cudaMemcpy(poly->vertexNorms, d_norms, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);
				cudaMemcpy(poly->vertexColorVals, d_vertexColorVals, sizeof(float)*poly->vertexcount, cudaMemcpyDeviceToHost);
			}

			cudaFree(d_errorEdgeInfo);
		}
	}

	cudaMemcpy(poly->vertexCoords, d_vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);

	if (isColoringDeformedPart)
	{
		cudaMemcpy(poly->vertexDeviateVals, d_vertexDeviateVals, sizeof(float)*poly->vertexcount, cudaMemcpyDeviceToHost);
	}
}

//////////////////////deform particle

struct functor_particleDeform_Cuboid
{
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


	functor_particleDeform_Cuboid( float3 _start, float3 _end, float _r, float _deformationScale, float _deformationScaleVertical, float3 _dir2nd)
		: start(_start), end(_end), r(_r), deformationScale(_deformationScale), deformationScaleVertical(_deformationScaleVertical), dir2nd(_dir2nd){}
};

struct functor_particleDeform_Circle
{

	float3 start, end;
	float r, radius;

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

			float3 prjPoint = start + l*tunnelVec;
			float dis = length(pos - prjPoint);
			if (dis < 0.0000001){
				pos = pos + cross(tunnelVec, make_float3(0, 0, 0.00001)) + cross(tunnelVec, make_float3(0, 0.00001, 0.1));//semi-random disturb
			}
			float3 dir = normalize(pos - prjPoint);
			if (dis < radius){
				float newDis = radius - (radius - dis) / radius * (radius - r);
				newPos = prjPoint + newDis * dir;

				float oneTimeThr = -1;;
				if (oneTimeThr > 0){
					if (length(newPos - pos) > oneTimeThr){
						newPos = pos + normalize(newPos - pos) * oneTimeThr;
					}
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

	functor_particleDeform_Circle(float3 _start, float3 _end, float _r, float _radius)
		:start(_start), end(_end), r(_r), radius(_radius){}
};

struct functor_particleDeform_Circle2
{
	float3 start, end;
	float r, radius;

	template<typename Tuple>
	__device__ __host__ void operator() (Tuple t){
		float4 posf4 = thrust::get<0>(t);
		float3 pos = make_float3(posf4.x, posf4.y, posf4.z);
		float3 newPos;

		float3 posLast = make_float3(thrust::get<1>(t));

		float3 tunnelVec = normalize(end - start);
		float tunnelLength = length(end - start);

		float3 voxelVec = pos - start;
		float l = dot(voxelVec, tunnelVec);
		if (l > 0 && l < tunnelLength){

			float3 prjPoint = start + l*tunnelVec;
			float dis = length(pos - prjPoint);
			if (dis < 0.0000001){
				pos = pos + cross(tunnelVec, make_float3(0, 0, 0.00001)) + cross(tunnelVec, make_float3(0, 0.00001, 0.1));//semi-random disturb
			}
			float3 dir = normalize(pos - prjPoint);
			if (dis < radius){
				float newDis = radius - (radius - dis) / radius * (radius - r);
				newPos = prjPoint + newDis * dir;

				float oneTimeThr = -1;
				if (oneTimeThr > 0){
					if (length(newPos - pos) > oneTimeThr){
						newPos = pos + normalize(newPos - pos) * oneTimeThr;
					}
				}

				float angleThr = 0.06;
				if (oneTimeThr > 0 && posLast.x > -500){
					float3 prjLast = start + dot(posLast - start, tunnelVec);
					float3 vecLast = normalize(posLast - prjLast);
					float ang = acos(dot(prjLast, vecLast));
					if (ang > angleThr){
						float3 rotateAxis = cross(vecLast, dir);
						float adjustAngle = -(ang - angleThr);  //rotate dir back for certain angle

						float rotateMat[9];
						float sinval = sin(adjustAngle), cosval = cos(adjustAngle);
						rotateMat[0] = cosval + rotateAxis.x*rotateAxis.x*(1 - cosval);
						rotateMat[1] = rotateAxis.x*rotateAxis.y*(1 - cosval) - rotateAxis.z*sinval;
						rotateMat[2] = rotateAxis.x*rotateAxis.z*(1 - cosval) + rotateAxis.y*sinval;
						rotateMat[3] = rotateAxis.x*rotateAxis.y*(1 - cosval) + rotateAxis.z*sinval;
						rotateMat[4] = cosval + rotateAxis.y*rotateAxis.y*(1 - cosval);
						rotateMat[5] = rotateAxis.y*rotateAxis.z*(1 - cosval) - rotateAxis.x*sinval;
						rotateMat[6] = rotateAxis.x*rotateAxis.z*(1 - cosval) - rotateAxis.y*sinval;
						rotateMat[7] = rotateAxis.y*rotateAxis.z*(1 - cosval) + rotateAxis.x*sinval;
						rotateMat[8] = cosval + rotateAxis.z*rotateAxis.z*(1 - cosval);

						float3 newDir = make_float3(rotateMat[0] * dir.x + rotateMat[1] * dir.y + rotateMat[2] * dir.z, 
													rotateMat[3] * dir.x + rotateMat[4] * dir.y + rotateMat[5] * dir.z, 
													rotateMat[6] * dir.x + rotateMat[7] * dir.y + rotateMat[8] * dir.z);

						newPos = prjPoint + newDis * newDir;
					}
				}

			}
			else{
				newPos = pos;
			}
		}
		else{
			newPos = pos;
		}
		thrust::get<2>(t) = make_float4(newPos.x, newPos.y, newPos.z, 1);

	}

	functor_particleDeform_Circle2(float3 _start, float3 _end, float _r, float _radius)
		:start(_start), end(_end), r(_r), radius(_radius){}
};


void PositionBasedDeformProcessor::doParticleDeform(float degree)
{
	if (!deformData)
		return;

	//int count = particle->numParticles;
	//for debug
	//	std::vector<float4> tt(count);
	//	//thrust::copy(tt.begin(), tt.end(), d_vec_posTarget.begin());
	//	std::cout << "pos of region 0 before: " << tt[0].x << " " << tt[0].y << " " << tt[0].z << std::endl;


	if (shapeModel == CUBOID){
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
			functor_particleDeform_Cuboid(tunnelStart, tunnelEnd, degree, deformationScale, deformationScaleVertical, rectVerticalDir));
	}
	else if (shapeModel == CIRCLE){
		//thrust::for_each(
		//	thrust::make_zip_iterator(
		//	thrust::make_tuple(
		//	d_vec_posOrig.begin(),
		//	d_vec_posTarget.begin()
		//	)),
		//	thrust::make_zip_iterator(
		//	thrust::make_tuple(
		//	d_vec_posOrig.end(),
		//	d_vec_posTarget.end()
		//	)),
		//	functor_particleDeform_Circle(tunnelStart, tunnelEnd, degree, radius));

		thrust::for_each(
			thrust::make_zip_iterator(
			thrust::make_tuple(
			d_vec_posOrig.begin(),
			d_vec_lastFramePos.begin(),
			d_vec_posTarget.begin()
			)),
			thrust::make_zip_iterator(
			thrust::make_tuple(
			d_vec_posOrig.end(),
			d_vec_lastFramePos.end(),
			d_vec_posTarget.end()
			)),
			functor_particleDeform_Circle2(tunnelStart, tunnelEnd, degree, radius));
	}

	thrust::copy(d_vec_posTarget.begin(), d_vec_posTarget.end(), &(particle->pos[0]));
	thrust::copy(d_vec_posTarget.begin(), d_vec_posTarget.end(), d_vec_lastFramePos.begin());

	//	std::cout << "moved particles by: " << degree <<" with count "<<count<< std::endl;
	//	std::cout << "pos of region 0: " << particle->pos[0].x << " " << particle->pos[0].y << " " << particle->pos[0].z << std::endl;
}

void PositionBasedDeformProcessor::getLastPos(std::vector<float4> &ret)
{
	ret.resize(d_vec_lastFramePos.size());
	thrust::copy(d_vec_lastFramePos.begin(), d_vec_lastFramePos.end(), &(ret[0]));
}

void PositionBasedDeformProcessor::newLastPos(std::vector<float4> &in)
{
	if (d_vec_lastFramePos.size() != in.size()){
		d_vec_lastFramePos.resize(in.size());
	}
	d_vec_lastFramePos.assign(&(in[0]), &(in[0]) + in.size());
}

void PositionBasedDeformProcessor::doParticleDeform2Tunnel(float degreeOpen, float degreeClose)
{
	if (!deformData)
		return;
	int count = particle->numParticles;

	//for debug
	//	std::vector<float4> tt(count);
	//	//thrust::copy(tt.begin(), tt.end(), d_vec_posTarget.begin());
	//	std::cout << "pos of region 0 before: " << tt[0].x << " " << tt[0].y << " " << tt[0].z << std::endl;

	thrust::device_vector<float4> d_vec_posMid(count);

	thrust::for_each(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		d_vec_posOrig.begin(),
		d_vec_posMid.begin()
		)),
		thrust::make_zip_iterator(
		thrust::make_tuple(
		d_vec_posOrig.end(),
		d_vec_posMid.end()
		)),
		functor_particleDeform_Cuboid(lastTunnelStart, lastTunnelEnd, degreeClose, deformationScale, deformationScaleVertical, lastDeformationDirVertical));


	thrust::for_each(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		d_vec_posMid.begin(),
		d_vec_posTarget.begin()
		)),
		thrust::make_zip_iterator(
		thrust::make_tuple(
		d_vec_posMid.end(),
		d_vec_posTarget.end()
		)),
		functor_particleDeform_Cuboid(tunnelStart, tunnelEnd, degreeOpen, deformationScale, deformationScaleVertical, rectVerticalDir));

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
		//doPolyDeform(r);  //currently simply use opening
	}
	else if (dataType == PARTICLE){
		doParticleDeform2Tunnel(r, rClose);

		//doParticleDeform(r);  //currently simply use opening
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
				r = finalDegree(); //reset r
			}
		}
		else if (systemState == DEFORMED){
		}
		else{
			std::cout << "STATE NOT DEFINED for force deform" << std::endl;
			exit(0);
		}
	}
	else if (false){//(tv){	//NOTE!! should not turn tv into true when at mixing state
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
				r = finalDegree(); //reset r
			}
			else{
				if (atProperLocation(eyeInLocal, true)){
					systemState = CLOSING;
					float passed = tunnelTimer1.getTime();
					tunnelTimer1.init(outTime, (passed >= outTime) ? 0 : (outTime - passed));
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
					systemState = OPENING;
					computeTunnelInfo(eyeInLocal);
					tunnelTimer1.init(outTime, 0);

					if (dataType == MESH){ //for poly data, the original data will be modified, which is not applicable to other types of data
						modifyPolyMesh();
					}
				}
			}
		}
		else if (systemState == DEFORMED){
			if (atProperLocation(eyeInLocal, true)){
				systemState = CLOSING;
				float passed = tunnelTimer1.getTime();
				tunnelTimer1.init(outTime, (passed >= outTime) ? 0 : (outTime - passed));
			}
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
				r = finalDegree(); //reset r
			}
			else{
				if (atProperLocation(eyeInLocal, true)){
					systemState = CLOSING;
					float passed = tunnelTimer1.getTime();
					tunnelTimer1.init(outTime, (passed >= outTime) ? 0 : (outTime - passed));
				}
				else if (!atProperLocation(eyeInLocal, false)){
					//if (dataType == VOLUME){ //currently only for volume data
						storeCurrentTunnel();
						computeTunnelInfo(eyeInLocal);
						systemState = MIXING;
						float passed = tunnelTimer1.getTime();
						tunnelTimer2.init(outTime, (passed >= outTime) ? 0 : (outTime - passed));
						tunnelTimer1.init(outTime, 0);
					//}
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
					//if (dataType == VOLUME){ //currently only for volume data
						storeCurrentTunnel();
						computeTunnelInfo(eyeInLocal);
						systemState = MIXING;
						//tunnelTimer2 = tunnelTimer1;//cannot assign in this way!!!
						tunnelTimer2.init(outTime, tunnelTimer1.getTime());
						tunnelTimer1.init(outTime, 0);
					//}
				}
			}
		}
		else if (systemState == MIXING){
			if (tunnelTimer1.out() && tunnelTimer2.out()){
				systemState = DEFORMED;
				tunnelTimer1.end();
				tunnelTimer2.end();
				r = finalDegree(); //reset r
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
				//if (dataType == VOLUME){ //currently only for volume data
					storeCurrentTunnel();
					computeTunnelInfo(eyeInLocal);
					systemState = MIXING;
					tunnelTimer2.init(outTime, 0);
					tunnelTimer1.init(outTime, 0);
				//}
			}
		}
		else{
			std::cout << "STATE NOT DEFINED" << std::endl;
			exit(0);
		}
	}

	if (systemState == MIXING){
		if (tunnelTimer2.out()){
			if (shapeModel == CUBOID){
				r = tunnelTimer1.getTime() / outTime * deformationScale / 2;
			}
			else if (shapeModel == CIRCLE){
				r = tunnelTimer1.getTime() / outTime * radius / 2;
			}
			deformDataByDegree(r);
			//std::cout << "doing mixinig with r: " << r << " and 0" << std::endl;
		}
		else{
			if (tunnelTimer1.out()){
				std::cout << "impossible combination!" << std::endl;
				exit(0);
			}
			else{
				if (shapeModel == CUBOID){
					rOpen = tunnelTimer1.getTime() / outTime * deformationScale / 2;
					rClose = (1 - tunnelTimer2.getTime() / outTime) * deformationScale / 2;
				}
				else if (shapeModel == CIRCLE){
					rOpen = tunnelTimer1.getTime() / outTime * radius / 2;
					rClose = (1 - tunnelTimer2.getTime() / outTime) * radius / 2;
				}		
				r = rOpen;//might be used elsewhere
				deformDataByDegree2Tunnel(rOpen, rClose);
				//std::cout << "doing mixinig with r: " << rOpen << " and " << rClose << std::endl;
			}
		}
	}
	else if (systemState == OPENING){
		if (shapeModel == CUBOID){
			r = tunnelTimer1.getTime() / outTime * deformationScale / 2;
		}
		else if (shapeModel == CIRCLE){
			r = tunnelTimer1.getTime() / outTime * radius / 2;
		}
		deformDataByDegree(r);
		//std::cout << "doing openning with r: " << r << std::endl;
	}
	else if (systemState == CLOSING){
		if (shapeModel == CUBOID){
			r = (1 - tunnelTimer1.getTime() / outTime) * deformationScale / 2;
		}
		else if (shapeModel == CIRCLE){
			r = (1 - tunnelTimer1.getTime() / outTime) * radius / 2;
		}
		deformDataByDegree(r);
		//std::cout << "doing closing with r: " << r << std::endl;
	}
	else if (systemState == DEFORMED){
		deformDataByDegree(r);
	}




	if (systemState != lastSystemState){
		std::cout << " current state " << systemState << std::endl;
	}
	lastSystemState = systemState;

	return true;
}






