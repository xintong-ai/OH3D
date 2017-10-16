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

//#include <cubicTex3D.cu>


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
	//cudaMemset(d_numAddedFaces, 0, sizeof(int));
}

void PositionBasedDeformProcessor::polyMeshDataUpdated()
{
	PrepareDataStructureForPolyDeform();
	if (systemState != ORIGINAL && isActive){
		modifyPolyMesh();
		doPolyDeform(r);
	}

	//!!!!!  NOT COMPLETE YET!!! to add more state changes!!!! for mixing
}

void PositionBasedDeformProcessor::volumeDataUpdated()
//only for changing rendering parameter
{
	if (systemState != ORIGINAL && isActive){
		//std::cout << "camera BAD in new original data" << std::endl;
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
			doVolumeDeform2Tunnel(rOpen, rClose);
		}
		else{
			doVolumeDeform(r); //although the deformation might be computed later, still do it once here to compute d_vec_posTarget in case of needed by detecting proper location
		}
	}
	else{
		//std::cout << "camera GOOD in new original data" << std::endl;
	}
}


PositionBasedDeformProcessor::PositionBasedDeformProcessor(std::shared_ptr<Particle> ori, std::shared_ptr<MatrixManager> _m)
{
	particle = ori;
	matrixMgr = _m;

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
	if (d_vec_lastFramePos.size() != particle->numParticles){
		d_vec_lastFramePos.resize(particle->numParticles);
	}
	if (particle->orientation.size() >0){
		d_vec_orientation.assign(&(particle->orientation[0]), &(particle->orientation[0]) + particle->numParticles);
	}
	else{
		std::cout << "WARNING! particles orientation expected but not found" << std::endl;
	}

	if (systemState != ORIGINAL && isActive){
		//std::cout << "camera BAD in new original data" << std::endl;


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
		//std::cout << "camera GOOD in new original data" << std::endl;
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
		step = 0.5;
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
			rectVerticalDir = matrixMgr->getUpInLocal();
		}

		tunnelStart = centerPoint;
		while (atProperLocation(tunnelStart, true)){
			tunnelStart += tunnelAxis*step;
		}
		tunnelEnd = tunnelStart + tunnelAxis*step;
		while (!atProperLocation(tunnelEnd, true)){
			tunnelEnd += tunnelAxis*step;
		}
	}
	else{
		//when this funciton is called, suppose we already know that centerPoint is inWall
		float3 tunnelAxis = normalize(matrixMgr->getViewVecInLocal());
		//rectVerticalDir = targetUpVecInLocal;
		if (abs(dot(targetUpVecInLocal, tunnelAxis)) < 0.9){
			rectVerticalDir = normalize(cross(cross(tunnelAxis, targetUpVecInLocal), tunnelAxis));
			//this one should also be the same with matrixMgr->getUpInLocal()
		}
		else{
			rectVerticalDir = matrixMgr->getUpInLocal();
		}

		tunnelEnd = centerPoint + tunnelAxis*step;
		while (!atProperLocation(tunnelEnd, true)){
			tunnelEnd += tunnelAxis*step;
		}
		tunnelStart = centerPoint;
		while (!atProperLocation(tunnelStart, true)){
			tunnelStart -= tunnelAxis*step;
		}
	}
}

void PositionBasedDeformProcessor::adjustTunnelEnds()
{
	float step;
	if (dataType == VOLUME){
		step = 1; //for VOLUME, no need to be less than 1
	}
	else if (dataType == PARTICLE){
		step = 0.5;
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

__global__ void d_checkIfTooCloseToPoly(float3 pos, uint* indices, int faceCoords, float* vertexCoords, float *norms, float thr, bool useDifThrForBack, bool* res)
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
	//float3 avenorm = (norm1 + norm2 + norm3) / 3;
	
	if (useDifThrForBack && dot(norm1, pos - v1) < 0 && dot(norm2, pos - v2) < 0 && dot(norm3, pos - v3) < 0){ //back side of the triangle
		if (dis < thr / 2)
		{
			*res = true;
		}
	}
	else{
		if (dis < thr)
		{
			*res = true;
		}
	}

	return;
}

bool PositionBasedDeformProcessor::inFullExtentTunnel(float3 pos)
{
	if (shapeModel == CUBOID){
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
	else if (shapeModel == CIRCLE){
		float3 tunnelVec = normalize(tunnelEnd - tunnelStart);
		float tunnelLength = length(tunnelEnd - tunnelStart);
		float3 voxelVec = pos - tunnelStart;
		float l = dot(voxelVec, tunnelVec);
		if (l >= 0 && l <= tunnelLength){
			float l3 = length(tunnelStart + tunnelVec * l - pos);
			if (abs(l3) < radius / 2){
				return true;
			}
		}
	}
	else{
		std::cout << "inFullExtentTunnel not implemented function !!!" << std::endl;
	}

	if (systemState == MIXING){
		if (shapeModel == CUBOID){
			float3 tunnelVec = normalize(lastTunnelEnd - lastTunnelStart);
			float tunnelLength = length(lastTunnelEnd - lastTunnelStart);
			float3 n = normalize(cross(lastDeformationDirVertical, tunnelVec));
			float3 voxelVec = pos - lastTunnelStart;
			float l = dot(voxelVec, tunnelVec);
			if (l >= 0 && l <= tunnelLength){
				float l2 = dot(voxelVec, lastDeformationDirVertical);
				if (abs(l2) < deformationScaleVertical){
					float l3 = dot(voxelVec, n);
					if (abs(l3) < deformationScale / 2){
						return true;
					}
				}
			}
		}
		else if (shapeModel == CIRCLE){
			float3 tunnelVec = normalize(lastTunnelEnd - lastTunnelStart);
			float tunnelLength = length(lastTunnelEnd - lastTunnelStart);
			float3 voxelVec = pos - lastTunnelStart;
			float l = dot(voxelVec, tunnelVec);
			if (l >= 0 && l <= tunnelLength){
				float l3 = length(lastTunnelStart + tunnelVec * l - pos);
				if (abs(l3) < radius / 2){
					return true;
				}
			}
		}
		else{
			std::cout << "inFullExtentTunnel not implemented function !!!" << std::endl;
		}
	}
	return false;
}

bool PositionBasedDeformProcessor::atProperLocation(float3 pos, bool useOriData)
{
	if (!inRange(pos)){
		return true;
	}

	if (!useOriData){
		if (systemState != ORIGINAL && systemState != CLOSING){
			//first check if inside the deform frame	
			//float3 tunnelVec = normalize(tunnelEnd - tunnelStart);
			//float tunnelLength = length(tunnelEnd - tunnelStart);
			//float3 n = normalize(cross(rectVerticalDir, tunnelVec));
			//float3 voxelVec = pos - tunnelStart;
			//float l = dot(voxelVec, tunnelVec);
			//if (l >= 0 && l <= tunnelLength){
			//	float l2 = dot(voxelVec, rectVerticalDir);
			//	if (abs(l2) < deformationScaleVertical){
			//		float l3 = dot(voxelVec, n);
			//		if (abs(l3) < deformationScale / 2){
			//			return true;
			//		}
			//	}
			//}

			if (inFullExtentTunnel(pos)){
				return true;
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
			d_checkIfTooCloseToPoly << <blocksPerGrid, threadsPerBlock >> >(pos, d_indices, poly->facecount, d_vertexCoords_init, d_norms, disThr, useDifThrForBack, d_tooCloseToData);
		}
		else{
			d_checkIfTooCloseToPoly << <blocksPerGrid, threadsPerBlock >> >(pos, d_indices, poly->facecount, d_vertexCoords, d_norms, disThr, useDifThrForBack, d_tooCloseToData);
		}

		bool tooCloseToData;
		cudaMemcpy(&tooCloseToData, d_tooCloseToData, sizeof(bool)* 1, cudaMemcpyDeviceToHost);
		cudaFree(d_tooCloseToData);
		return !tooCloseToData;
	}
	else if (dataType == PARTICLE){
		float init = 10000;
		float inSavePosition;

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
				functor_ParticleDis(pos, disThrOriented[0], disThrOriented[1]));
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
				functor_ParticleDis(pos, disThrOriented[0], disThrOriented[1]));
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

__global__ void d_modifyMeshKernel_CuboidModel(float* vertexCoords, unsigned int* indices, int facecount, int vertexcount, float* norms, float3 start, float3 end, float r, float deformationScale, float deformationScaleVertical, float3 dir2nd, int* numAddedFaces, float* vertexColorVals, int* futureEdges, int* numFutureEdges)
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
	bool planeIntersectLong1 = projLength1long > 0 && projLength1long < tunnelLength ;
	bool planeIntersectLong2 = projLength2long > 0 && projLength2long < tunnelLength;
	bool planeIntersectShort1 = abs(projLength1short) < deformationScaleVertical;
	bool planeIntersectShort2 = abs(projLength2short) < deformationScaleVertical;
	if (planeIntersectLong1 && planeIntersectLong2 && (planeIntersectShort1 || planeIntersectShort2)){
		indices[3 * i] = 0;
		indices[3 * i + 1] = 0;
		indices[3 * i + 2] = 0;

		int numAddedFacesBefore = atomicAdd(numAddedFaces, 3); //each divided triangle creates 3 new faces

		int curNumVertex = vertexcount + 4 * numAddedFacesBefore / 3; //each divided triangle creates 4 new vertex
		if (planeIntersectShort1){
			vertexCoords[3 * curNumVertex] = intersect1.x + disturb.x;
			vertexCoords[3 * curNumVertex + 1] = intersect1.y + disturb.y;
			vertexCoords[3 * curNumVertex + 2] = intersect1.z + disturb.z;
			vertexCoords[3 * (curNumVertex + 2)] = intersect1.x - disturb.x;
			vertexCoords[3 * (curNumVertex + 2) + 1] = intersect1.y - disturb.y;
			vertexCoords[3 * (curNumVertex + 2) + 2] = intersect1.z - disturb.z;
		}
		else{
			vertexCoords[3 * curNumVertex] = intersect1.x;
			vertexCoords[3 * curNumVertex + 1] = intersect1.y;
			vertexCoords[3 * curNumVertex + 2] = intersect1.z;
			vertexCoords[3 * (curNumVertex + 2)] = intersect1.x;
			vertexCoords[3 * (curNumVertex + 2) + 1] = intersect1.y;
			vertexCoords[3 * (curNumVertex + 2) + 2] = intersect1.z;

			int numFutureEdgesBefore = atomicAdd(numFutureEdges, 1);
			futureEdges[4 * numFutureEdgesBefore] = bottomV1;
			futureEdges[4 * numFutureEdgesBefore + 1] = separateVectex;
			futureEdges[4 * numFutureEdgesBefore + 2] = curNumVertex;
			futureEdges[4 * numFutureEdgesBefore + 3] = curNumVertex + 2;
		}
		if (planeIntersectShort2){
			vertexCoords[3 * (curNumVertex + 1)] = intersect2.x + disturb.x;
			vertexCoords[3 * (curNumVertex + 1) + 1] = intersect2.y + disturb.y;
			vertexCoords[3 * (curNumVertex + 1) + 2] = intersect2.z + disturb.z;
			vertexCoords[3 * (curNumVertex + 3)] = intersect2.x - disturb.x;
			vertexCoords[3 * (curNumVertex + 3) + 1] = intersect2.y - disturb.y;
			vertexCoords[3 * (curNumVertex + 3) + 2] = intersect2.z - disturb.z;
		}
		else{
			vertexCoords[3 * (curNumVertex + 1)] = intersect2.x;
			vertexCoords[3 * (curNumVertex + 1) + 1] = intersect2.y;
			vertexCoords[3 * (curNumVertex + 1) + 2] = intersect2.z;
			vertexCoords[3 * (curNumVertex + 3)] = intersect2.x;
			vertexCoords[3 * (curNumVertex + 3) + 1] = intersect2.y;
			vertexCoords[3 * (curNumVertex + 3) + 2] = intersect2.z;

			int numFutureEdgesBefore = atomicAdd(numFutureEdges, 1);
			futureEdges[4 * numFutureEdgesBefore] = separateVectex;
			futureEdges[4 * numFutureEdgesBefore + 1] = bottomV2;
			futureEdges[4 * numFutureEdgesBefore + 2] = curNumVertex + 1;
			futureEdges[4 * numFutureEdgesBefore + 3] = curNumVertex + 3;
		}

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

__global__ void d_modifyMeshKernel_CuboidModel_round2(unsigned int* indices, int facecount, int* numAddedFaces, int* futureEdges, int* numFutureEdges)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= facecount)	return;


	uint3 inds = make_uint3(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);


	for (int i = 0; i < *numFutureEdges; i++){
		int bottomV1 = futureEdges[4 * i];
		int bottomV2 = futureEdges[4 * i + 1];
		if ((inds.x == bottomV1 && inds.y == bottomV2) || (inds.y == bottomV1 && inds.x == bottomV2)){
			int numAddedFacesBefore = atomicAdd(numAddedFaces, 2); //each divided triangle creates 3 new faces
			int curNumFaces = numAddedFacesBefore + facecount;

			indices[3 * curNumFaces] = inds.x;
			indices[3 * curNumFaces + 1] = futureEdges[4 * i + 2];  //order of vertex matters! use counter clockwise
			indices[3 * curNumFaces + 2] = inds.z;
			indices[3 * (curNumFaces + 1)] = futureEdges[4 * i + 3];
			indices[3 * (curNumFaces + 1) + 1] = inds.y;
			indices[3 * (curNumFaces + 1) + 2] = inds.z;

			indices[3 * i] = 0;
			indices[3 * i + 1] = 0;
			indices[3 * i + 2] = 0;
		}
		else if ((inds.x == bottomV1 && inds.z == bottomV2) || (inds.z == bottomV1 && inds.x == bottomV2)){
			int numAddedFacesBefore = atomicAdd(numAddedFaces, 2); //each divided triangle creates 3 new faces
			int curNumFaces = numAddedFacesBefore + facecount;

			indices[3 * curNumFaces] = inds.x;
			indices[3 * curNumFaces + 1] = inds.y;  //order of vertex matters! use counter clockwise
			indices[3 * curNumFaces + 2] = futureEdges[4 * i + 2];
			indices[3 * (curNumFaces + 1)] = futureEdges[4 * i + 3];
			indices[3 * (curNumFaces + 1) + 1] = inds.y;
			indices[3 * (curNumFaces + 1) + 2] = inds.z;

			indices[3 * i] = 0;
			indices[3 * i + 1] = 0;
			indices[3 * i + 2] = 0;
		}
		else if ((inds.y == bottomV1 && inds.z == bottomV2) || (inds.z == bottomV1 && inds.y == bottomV2)){
			int numAddedFacesBefore = atomicAdd(numAddedFaces, 2); //each divided triangle creates 3 new faces
			int curNumFaces = numAddedFacesBefore + facecount;

			indices[3 * curNumFaces] = inds.x;
			indices[3 * curNumFaces + 1] = inds.y;  //order of vertex matters! use counter clockwise
			indices[3 * curNumFaces + 2] = futureEdges[4 * i + 2];
			indices[3 * (curNumFaces + 1)] = inds.x;
			indices[3 * (curNumFaces + 1) + 1] = futureEdges[4 * i + 3];
			indices[3 * (curNumFaces + 1) + 2] = inds.z;

			indices[3 * i] = 0;
			indices[3 * i + 1] = 0;
			indices[3 * i + 2] = 0;
		}

	}
}


void PositionBasedDeformProcessor::modifyPolyMesh()
{
	cudaMemcpy(d_vertexCoords, d_vertexCoords_init, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost); //need to do this since for mixing, d_vertexCoords may be dif from d_vertexCoords_init
	cudaMemcpy(d_indices, d_indices_init, sizeof(float)*poly->facecount * 3, cudaMemcpyDeviceToHost); //need to do this since for mixing, d_indices may be dif from d_indices_init

	if (shapeModel == CUBOID){
		int threadsPerBlock = 64;
		int blocksPerGrid = (poly->vertexcount + threadsPerBlock - 1) / threadsPerBlock;
		d_disturbVertex_CuboidModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, poly->vertexcount,
			tunnelStart, tunnelEnd, deformationScaleVertical, rectVerticalDir);
	}
	else if (shapeModel == CIRCLE){
		std::cout << "circle model for poly not implemented!! " << std::endl;
		return;
	}

	int numAddedFaces;
	int numAddedVertices;
	cudaMemset(d_numAddedFaces, 0, sizeof(int));

	if (shapeModel == CUBOID){
		int threadsPerBlock = 64;
		int blocksPerGrid = (poly->facecount + threadsPerBlock - 1) / threadsPerBlock;

		int maxFutureEdgesSupported = 12;
		int* d_futureEdges = 0;
		cudaMalloc(&d_futureEdges, sizeof(int)* maxFutureEdgesSupported);
		int* d_numFutureEdges;
		cudaMalloc(&d_numFutureEdges, sizeof(int));
		cudaMemset(d_numFutureEdges, 0, sizeof(int));


		d_modifyMeshKernel_CuboidModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, d_indices, poly->facecount, poly->vertexcount, d_norms,
			tunnelStart, tunnelEnd, deformationScale, deformationScale, deformationScaleVertical, rectVerticalDir,
			d_numAddedFaces, d_vertexColorVals, d_futureEdges, d_numFutureEdges);

		cudaMemcpy(&numAddedFaces, d_numAddedFaces, sizeof(int), cudaMemcpyDeviceToHost);
		numAddedVertices = numAddedFaces / 3 * 4;


		int tt;
		cudaMemcpy(&tt, d_numFutureEdges, sizeof(int), cudaMemcpyDeviceToHost);
		//std::cout << "future edge to do count: " << tt << std::endl;
		if (tt > maxFutureEdgesSupported){
			std::cout << "!!!! unexpected count of future edge to process: " << tt << std::endl;
		}

		d_modifyMeshKernel_CuboidModel_round2 << <blocksPerGrid, threadsPerBlock >> >(d_indices, poly->facecount, d_numAddedFaces, d_futureEdges, d_numFutureEdges);

		//a few new faces added again
		cudaMemcpy(&numAddedFaces, d_numAddedFaces, sizeof(int), cudaMemcpyDeviceToHost);
		//numAddedVertices = numAddedFaces / 3 * 4;

		cudaFree(d_futureEdges);
		cudaFree(d_numFutureEdges);
	}

	int oldf = poly->facecount, oldv = poly->vertexcount;
	poly->facecount += numAddedFaces;
	poly->vertexcount += numAddedVertices;

	//std::cout << "number of face count " << oldf << " -> " << poly->facecount << std::endl;
	//std::cout << "number of vertex count " << oldv << " -> " << poly->vertexcount << std::endl;

	cudaMemcpy(poly->indices, d_indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(poly->vertexCoords, d_vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(poly->vertexNorms, d_norms, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);
	cudaMemcpy(poly->vertexColorVals, d_vertexColorVals, sizeof(float)*poly->vertexcount, cudaMemcpyDeviceToHost);

	cudaMemcpy(d_vertexCoords_init, d_vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_indices_init, d_indices, sizeof(unsigned int)*poly->facecount * 3, cudaMemcpyDeviceToDevice);
}

void PositionBasedDeformProcessor::modifyPolyMeshByAddingOneTunnel()
{
	//the modifyPolyMesh() function can successfully modify the vert and indices to add info of the new tunnel, so can be reused directly here when change from open/close to mix
	modifyPolyMesh();
}


void PositionBasedDeformProcessor::resetToOneTunnelStructure()//when state changes from mix to defomred
{
	if (shapeModel == CIRCLE){
		std::cout << "circle model for poly not implemented!! " << std::endl;
		return;
	}

	//easiest but may not be the most efficient
	resetData();

	////first reset everything back to before
	//poly->vertexcount = poly->vertexcountOri;
	//poly->facecount = poly->facecountOri;
	//cudaMemcpy(d_vertexCoords_init, poly->vertexCoordsOri, sizeof(float)*poly->vertexcount * 3, cudaMemcpyHostToDevice); //need to do this since for mixing, d_vertexCoords may be dif from d_vertexCoords_init
	//cudaMemcpy(d_indices_init, poly->indicesOri, sizeof(float)*poly->facecount * 3, cudaMemcpyHostToDevice); //need to do this since for mixing, d_vertexCoords may be dif from d_vertexCoords_init

	//modify mesh for tunnel1
	modifyPolyMesh();

	////modify mesh for tunnel2
	//float3 tstart = tunnelStart;
	//float3 tend = tunnelEnd;
	//float3 tvirt = rectVerticalDir;
	//tunnelStart = lastTunnelStart;
	//tunnelEnd = lastTunnelEnd;
	//rectVerticalDir = lastDeformationDirVertical;
	//modifyPolyMesh();
	//tunnelStart = tstart;
	//tunnelEnd = tend;
	//rectVerticalDir = tvirt;
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

__global__ void d_deformPolyMesh_ComputeDeviate(float* vertexCoords_init, float* vertexCoords, int vertexcount, float deformationScale, float* vertexDeviateVals)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= vertexcount)	return;


	float3 pos = make_float3(vertexCoords_init[3 * i], vertexCoords_init[3 * i + 1], vertexCoords_init[3 * i + 2]);

	float3 newPos = make_float3(vertexCoords[3 * i], vertexCoords[3 * i + 1], vertexCoords[3 * i + 2]);

	vertexDeviateVals[i] = length(newPos - pos) / (deformationScale / 2);

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
		std::cout << "not implemented for circle model of poly data!" << std::endl;
		return;
	}

	cudaMemcpy(poly->vertexCoords, d_vertexCoords, sizeof(float)*poly->vertexcount * 3, cudaMemcpyDeviceToHost);

	if (isColoringDeformedPart)
	{
		cudaMemcpy(poly->vertexDeviateVals, d_vertexDeviateVals, sizeof(float)*poly->vertexcount, cudaMemcpyDeviceToHost);
	}
}

void PositionBasedDeformProcessor::doPolyDeform2Tunnel(float degreeOpen, float degreeClose)
{
	if (!deformData)
		return;
	int threadsPerBlock = 64;
	int blocksPerGrid = (poly->vertexcount + threadsPerBlock - 1) / threadsPerBlock;

	if (shapeModel == CUBOID){
		d_deformPolyMesh_CuboidModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords_init, d_vertexCoords, poly->vertexcount, lastTunnelStart, lastTunnelEnd, degreeClose, deformationScale, deformationScaleVertical, lastDeformationDirVertical, d_vertexDeviateVals);
		d_deformPolyMesh_CuboidModel << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords, d_vertexCoords, poly->vertexcount, tunnelStart, tunnelEnd, degreeOpen, deformationScale, deformationScaleVertical, rectVerticalDir, d_vertexDeviateVals);

		d_deformPolyMesh_ComputeDeviate << <blocksPerGrid, threadsPerBlock >> >(d_vertexCoords_init, d_vertexCoords, poly->vertexcount, deformationScale, d_vertexDeviateVals);
	}
	else if (shapeModel == CIRCLE){
		std::cout << "not implemented for circle model of poly data!" << std::endl;
		return;
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


	functor_particleDeform_Cuboid(float3 _start, float3 _end, float _r, float _deformationScale, float _deformationScaleVertical, float3 _dir2nd)
		: start(_start), end(_end), r(_r), deformationScale(_deformationScale), deformationScaleVertical(_deformationScaleVertical), dir2nd(_dir2nd){}
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

				float angleThr = 0.04;
				if (angleThr > 0 && posLast.x > -500){

					float3 prjLast = start + dot(posLast - start, tunnelVec) * tunnelVec;
					float3 vecLast = normalize(posLast - prjLast);
					float ang = acosf(dot(dir, vecLast));
					if (ang > angleThr){
						float3 rotateAxisPre = cross(vecLast, dir);
						float3 rotateAxis;
						if (length(rotateAxisPre) < 0.0001){
							rotateAxis = -tunnelVec;
						}
						else{
							rotateAxis = normalize(rotateAxisPre);
						}
						float adjustAngle = -(ang - angleThr);  //rotate dir back for certain angle

						float rotateMat[9];
						float sinval = sinf(adjustAngle), cosval = cosf(adjustAngle);
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

					//float3 prjLast = start + dot(posLast - start, tunnelVec);
					//float3 vecLast = normalize(posLast - prjLast);
					//float ang = acos(dot(prjLast, vecLast));
					//if (ang > angleThr){
					//	float3 rotateAxis = cross(vecLast, dir);
					//	float adjustAngle = -(ang - angleThr);  //rotate dir back for certain angle

					//	float rotateMat[9];
					//	float sinval = sin(adjustAngle), cosval = cos(adjustAngle);
					//	rotateMat[0] = cosval + rotateAxis.x*rotateAxis.x*(1 - cosval);
					//	rotateMat[1] = rotateAxis.x*rotateAxis.y*(1 - cosval) - rotateAxis.z*sinval;
					//	rotateMat[2] = rotateAxis.x*rotateAxis.z*(1 - cosval) + rotateAxis.y*sinval;
					//	rotateMat[3] = rotateAxis.x*rotateAxis.y*(1 - cosval) + rotateAxis.z*sinval;
					//	rotateMat[4] = cosval + rotateAxis.y*rotateAxis.y*(1 - cosval);
					//	rotateMat[5] = rotateAxis.y*rotateAxis.z*(1 - cosval) - rotateAxis.x*sinval;
					//	rotateMat[6] = rotateAxis.x*rotateAxis.z*(1 - cosval) - rotateAxis.y*sinval;
					//	rotateMat[7] = rotateAxis.y*rotateAxis.z*(1 - cosval) + rotateAxis.x*sinval;
					//	rotateMat[8] = cosval + rotateAxis.z*rotateAxis.z*(1 - cosval);

					//	float3 newDir = make_float3(rotateMat[0] * dir.x + rotateMat[1] * dir.y + rotateMat[2] * dir.z, 
					//								rotateMat[3] * dir.x + rotateMat[4] * dir.y + rotateMat[5] * dir.z, 
					//								rotateMat[6] * dir.x + rotateMat[7] * dir.y + rotateMat[8] * dir.z);

					//	newPos = prjPoint + newDis * newDir;
					//}
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

// Inline calculation of the bspline convolution weights, without conditional statements
inline __device__ void WEIGHTS_my(float3 fraction, float3& w0, float3& w1, float3& w2, float3& w3)
{
	const float3 one_frac = 1.0f - fraction;
	const float3 squared = fraction * fraction;
	const float3 one_sqd = one_frac * one_frac;

	w0 = 1.0f / 6.0f * one_sqd * one_frac;
	w1 = 2.0f / 3.0f - 0.5f * squared * (2.0f - fraction);
	w2 = 2.0f / 3.0f - 0.5f * one_sqd * (2.0f - one_frac);
	w3 = 1.0f / 6.0f * squared * fraction;
}

__device__ float CUBICTEX3D_my(float3 coord)
{
	// shift the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float3 coord_grid = coord - 0.5f;
	const float3 index = floorf(coord_grid);
	const float3 fraction = coord_grid - index;
	float3 w0, w1, w2, w3;
	WEIGHTS_my(fraction, w0, w1, w2, w3);

	const float3 g0 = w0 + w1;
	const float3 g1 = w2 + w3;
	const float3 h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
	const float3 h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

	// fetch the eight linear interpolations
	// weighting and fetching is interleaved for performance and stability reasons
	float tex000 = tex3D(volumeTexInput, h0.x, h0.y, h0.z);
	float tex100 = tex3D(volumeTexInput, h1.x, h0.y, h0.z);
	tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
	float tex010 = tex3D(volumeTexInput, h0.x, h1.y, h0.z);
	float tex110 = tex3D(volumeTexInput, h1.x, h1.y, h0.z);
	tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
	tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
	float tex001 = tex3D(volumeTexInput, h0.x, h0.y, h1.z);
	float tex101 = tex3D(volumeTexInput, h1.x, h0.y, h1.z);
	tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
	float tex011 = tex3D(volumeTexInput, h0.x, h1.y, h1.z);
	float tex111 = tex3D(volumeTexInput, h1.x, h1.y, h1.z);
	tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
	tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction

	return (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction
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
				//sample = cubicTex3D(volumeTexValueForRC, coord.x, coord.y, coord.z);
				res = CUBICTEX3D_my(make_float3(samplePos.x + 0.5, samplePos.y + 0.5, samplePos.z + 0.5));

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
		doPolyDeform2Tunnel(r, rClose);
	}
	else if (dataType == PARTICLE){
		doParticleDeform2Tunnel(r, rClose);
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

	//if (isForceDeform && r > 0.05) //used to draw the triangle cutting of poly meshes
	//return false;

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
					storeCurrentTunnel();
					computeTunnelInfo(eyeInLocal);
					systemState = MIXING;
					float passed = tunnelTimer1.getTime();
					tunnelTimer2.init(outTime, (passed >= outTime) ? 0 : (outTime - passed));
					tunnelTimer1.init(outTime, 0);

					if (dataType == MESH){ //for poly data, the original data will be modified, which is not applicable to other types of data
						modifyPolyMeshByAddingOneTunnel();
					}
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

					if (dataType == MESH){ //for poly data, the original data will be modified, which is not applicable to other types of data
						modifyPolyMeshByAddingOneTunnel();
					}
				}
			}
		}
		else if (systemState == MIXING){
			if (tunnelTimer1.out() && tunnelTimer2.out()){
				systemState = DEFORMED;
				tunnelTimer1.end();
				tunnelTimer2.end();
				r = finalDegree(); //reset r
				if (dataType == MESH){ //for poly data, when state in mix, modification of mesh contains 2 tunnels. now only 1 is needed
					resetToOneTunnelStructure();
				}	
			}
			else if (tunnelTimer2.out()){
				systemState = OPENING;
				tunnelTimer2.end();
				if (dataType == MESH){ //for poly data, when state in mix, modification of mesh contains 2 tunnels. now only 1 is needed
					resetToOneTunnelStructure();
				}
			}
			else if (tunnelTimer1.out()){
				std::cout << "impossible combination!" << std::endl;
				exit(0);
			}
			else if (atProperLocation(eyeInLocal, true)){
				//tunnelTimer2 may not have been out() yet, but here ignore the 2nd tunnel. i.e., now we do not process double closing
				systemState = CLOSING;
				float passed = tunnelTimer1.getTime();
				tunnelTimer1.init(outTime, (passed >= outTime) ? 0 : (outTime - passed));
			}
			else if (atProperLocation(eyeInLocal, false)){
				//proper in current, no need to do anything
			}
			else{ //mixing of 2 different tunnels

				if (dataType == MESH){ //first remove the old tunnel that is unuseful now
					resetToOneTunnelStructure();
				}
				storeCurrentTunnel();
				computeTunnelInfo(eyeInLocal);
				float passed = tunnelTimer1.getTime();
				tunnelTimer2.init(outTime, (passed >= outTime) ? 0 : (outTime - passed));
				tunnelTimer1.init(outTime, 0);

				if (dataType == MESH){ //the add the new tunnel
					modifyPolyMeshByAddingOneTunnel();
				}
				std::cout << " current state NEW MIXING" << std::endl;
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

				if (dataType == MESH){ //for poly data, the original data will be modified, which is not applicable to other types of data
					modifyPolyMeshByAddingOneTunnel();
				}
			}
		}
		else{
			std::cout << "STATE NOT DEFINED" << std::endl;
			exit(0);
		}
	}

	if (systemState == MIXING){
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


		//if (tunnelTimer2.out()){
		//	if (shapeModel == CUBOID){
		//		r = tunnelTimer1.getTime() / outTime * deformationScale / 2;
		//	}
		//	else if (shapeModel == CIRCLE){
		//		r = tunnelTimer1.getTime() / outTime * radius / 2;
		//	}
		//	deformDataByDegree(r);
		//	//std::cout << "doing mixinig with r: " << r << " and 0" << std::endl;
		//}
		//else{
		//	if (tunnelTimer1.out()){
		//		std::cout << "impossible combination!" << std::endl;
		//		exit(0);
		//	}
		//	else{
		//		if (shapeModel == CUBOID){
		//			rOpen = tunnelTimer1.getTime() / outTime * deformationScale / 2;
		//			rClose = (1 - tunnelTimer2.getTime() / outTime) * deformationScale / 2;
		//		}
		//		else if (shapeModel == CIRCLE){
		//			rOpen = tunnelTimer1.getTime() / outTime * radius / 2;
		//			rClose = (1 - tunnelTimer2.getTime() / outTime) * radius / 2;
		//		}
		//		r = rOpen;//might be used elsewhere
		//		deformDataByDegree2Tunnel(rOpen, rClose);
		//		//std::cout << "doing mixinig with r: " << rOpen << " and " << rClose << std::endl;
		//	}
		//}
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
		printState();
	}
	lastSystemState = systemState;

	return true;
}



void PositionBasedDeformProcessor::printState()
{
	switch (systemState)
	{
	case ORIGINAL:
		std::cout << " current state ORIGINAL" << std::endl;
		break;
	case DEFORMED:
		std::cout << " current state DEFORMED" << std::endl;
		break;
	case OPENING:
		std::cout << " current state OPENING" << std::endl;
		break;
	case CLOSING:
		std::cout << " current state CLOSING" << std::endl;
		break;
	case MIXING:
		std::cout << " current state MIXING" << std::endl;
		break;
	default:
		break;
	}
}