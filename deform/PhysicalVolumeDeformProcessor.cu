#include "PhysicalVolumeDeformProcessor.h"
#include "Lens.h"
#include "MeshDeformProcessor.h"
#include "Volume.h"
#include "TransformFunc.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>



texture<float, 3, cudaReadModeElementType>  volumeTexInput;
surface<void, cudaSurfaceType3D>			volumeSurfaceOut;


void ModelVolumeDeformer_KernelInit()
{
	volumeTexInput.normalized = false;
	volumeTexInput.filterMode = cudaFilterModeLinear;
	volumeTexInput.addressMode[0] = cudaAddressModeBorder;
	volumeTexInput.addressMode[1] = cudaAddressModeBorder;
	volumeTexInput.addressMode[2] = cudaAddressModeBorder;
}


void PhysicalVolumeDeformProcessor::InitFromVolume(std::shared_ptr<Volume> _ori)
{
	volume = _ori;

	//volumeCUDADeformed.VolumeCUDA_init(_ori->size, _ori->values, 1, 1);

	ModelVolumeDeformer_KernelInit();
}



__global__ void
d_updateVolumebyModelGrid_init(cudaExtent volumeSize, float3 lensSpaceOrigin, float3 majorAxis, float3 minorAxis, float3 lensDir, float3 range, float3 spacing)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
	{
		return;
	}					
	
	float3 v1 = make_float3(x, y, z) * spacing - lensSpaceOrigin;
	float xInLensSpace = dot(v1, majorAxis);
	float yInLensSpace = dot(v1, minorAxis);
	float zInLensSpace = dot(v1, lensDir);
	if (xInLensSpace <= 0 || xInLensSpace >= range.x || yInLensSpace <= 0 || yInLensSpace >= range.y || zInLensSpace <= 0 || zInLensSpace >= range.z){
		float res = tex3D(volumeTexInput, x + 0.5, y + 0.5, z + 0.5);
		surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
	}
	else{
		float res = 0;
		surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
	}
	return;
}



__global__ void 
d_updateVolumebyModelGridOld(cudaExtent volumeSize, const float* X, const float* X_Orig, const int* tet, int tet_number, float3 spacing)
{	
	int tetId = blockIdx.x;

	//int tetId = blockIdx.x*blockDim.x + threadIdx.x;
	//if (tetId >= tet_number)return;

	float3 boundmin = make_float3(9999, 9999, 9999), boundmax = make_float3(-9999, -9999, -9999);
	float3 verts[4];
	float3 vertsOri[4];
	for (int v = 0; v < 4; v++){
		int vertInd = tet[4 * tetId + v];

		verts[v] = make_float3(X[3 * vertInd], X[3 * vertInd + 1], X[3 * vertInd + 2]);
		vertsOri[v] = make_float3(X_Orig[3 * vertInd], X_Orig[3 * vertInd + 1], X_Orig[3 * vertInd + 2]);

		boundmin = fminf(boundmin, verts[v]);
		boundmax = fmaxf(boundmax, verts[v]);
	}

	float3 volumeBoundmin = boundmin / spacing, volumeBoundmax = boundmax / spacing;
	
	int zPieceSize = (floor(volumeBoundmax.z) - ceil(volumeBoundmin.z) + 1 + blockDim.z - 1) / blockDim.z;
	int zstart = ceil(volumeBoundmin.z) + zPieceSize*threadIdx.z, zend = min(zstart + zPieceSize - 1, (int)floor(volumeBoundmax.z));
	int yPieceSize = (floor(volumeBoundmax.y) - ceil(volumeBoundmin.y) + 1 + blockDim.y - 1) / blockDim.y;
	int ystart = ceil(volumeBoundmin.y) + yPieceSize*threadIdx.y, yend = min(ystart + yPieceSize - 1, (int)floor(volumeBoundmax.y));
	int xPieceSize = (floor(volumeBoundmax.x) - ceil(volumeBoundmin.x) + 1 + blockDim.x - 1) / blockDim.x;
	int xstart = ceil(volumeBoundmin.x) + xPieceSize*threadIdx.x, xend = min(xstart + xPieceSize - 1, (int)floor(volumeBoundmax.x));

	if (xend < 0 || yend < 0 || zend < 0 ||
		xstart >= volumeSize.width || ystart >= volumeSize.height || zstart >= volumeSize.depth){
		return;
	}

	//colume-major matrix
	float matB2C[16] = { verts[0].x, verts[0].y, verts[0].z,1.0,
		verts[1].x, verts[1].y, verts[1].z, 1.0,
		verts[2].x, verts[2].y, verts[2].z, 1.0, 
		verts[3].x, verts[3].y, verts[3].z, 1.0 };//baricentric coord 2 cartisan coord

	float matC2B[16];
	if (!invertMatrix(matB2C, matC2B))
		return;

	for (int z = zstart; z <= zend; z++){
		for (int y = ystart; y <= yend; y++){
			for (int x = xstart; x <= xend; x++){
				if (x >= 0 && y >= 0 && z >= 0 &&
					x < volumeSize.width && y < volumeSize.height && z < volumeSize.depth)
				{
					float3 x1 = make_float3(x, y, z) * spacing;
					float4 b = mat4mulvec4(matC2B, make_float4(x1, 1.0));
				
					if (b.x >= 0 && b.x <= 1
						&& b.y >= 0 && b.y <= 1
						&& b.z >= 0 && b.z <= 1
						&& b.w >= 0 && b.w <= 1){

						float3 x0 = b.x * vertsOri[0] + b.y * vertsOri[1] + b.z * vertsOri[2] + b.w * vertsOri[3];
						x0 = x0 / spacing;
						float res = tex3D(volumeTexInput, x0.x + 0.5, x0.y + 0.5, x0.z + 0.5);

						//float4 res = make_float4(0, 0, 0, 1.0*tetId / 450);
						surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
					}
				}
			}
		}
	}
}

//do a normalization of the 4 tet vertices before computing the barycentric coordinate 
__global__ void
d_updateVolumebyModelGrid(cudaExtent volumeSize, const float* X, const float* X_Orig, const int* tet, int tet_number, float3 spacing)
{
	int tetId = blockIdx.x;

	//int tetId = blockIdx.x*blockDim.x + threadIdx.x;
	//if (tetId >= tet_number)return;

	float3 boundmin = make_float3(9999, 9999, 9999), boundmax = make_float3(-9999, -9999, -9999);
	float3 verts[4];
	float3 vertsOri[4];
	for (int v = 0; v < 4; v++){
		int vertInd = tet[4 * tetId + v];

		verts[v] = make_float3(X[3 * vertInd], X[3 * vertInd + 1], X[3 * vertInd + 2]);
		vertsOri[v] = make_float3(X_Orig[3 * vertInd], X_Orig[3 * vertInd + 1], X_Orig[3 * vertInd + 2]);

		boundmin = fminf(boundmin, verts[v]);
		boundmax = fmaxf(boundmax, verts[v]);
	}

	float3 volumeBoundmin = boundmin / spacing, volumeBoundmax = boundmax / spacing;

	int zPieceSize = (floor(volumeBoundmax.z) - ceil(volumeBoundmin.z) + 1 + blockDim.z - 1) / blockDim.z;
	int zstart = ceil(volumeBoundmin.z) + zPieceSize*threadIdx.z, zend = min(zstart + zPieceSize - 1, (int)floor(volumeBoundmax.z));
	int yPieceSize = (floor(volumeBoundmax.y) - ceil(volumeBoundmin.y) + 1 + blockDim.y - 1) / blockDim.y;
	int ystart = ceil(volumeBoundmin.y) + yPieceSize*threadIdx.y, yend = min(ystart + yPieceSize - 1, (int)floor(volumeBoundmax.y));
	int xPieceSize = (floor(volumeBoundmax.x) - ceil(volumeBoundmin.x) + 1 + blockDim.x - 1) / blockDim.x;
	int xstart = ceil(volumeBoundmin.x) + xPieceSize*threadIdx.x, xend = min(xstart + xPieceSize - 1, (int)floor(volumeBoundmax.x));

	if (xend < 0 || yend < 0 || zend < 0 ||
		xstart >= volumeSize.width || ystart >= volumeSize.height || zstart >= volumeSize.depth){
		return;
	}


	float3 ave = (verts[0] + verts[1] + verts[2] + verts[3]) / 4;
	verts[0] -= ave;
	verts[1] -= ave;
	verts[2] -= ave;
	verts[3] -= ave;
	//colume-major matrix
	float matB2C[16] = { verts[0].x, verts[0].y, verts[0].z, 1.0,
		verts[1].x, verts[1].y, verts[1].z, 1.0,
		verts[2].x, verts[2].y, verts[2].z, 1.0,
		verts[3].x, verts[3].y, verts[3].z, 1.0 };//baricentric coord 2 cartisan coord

	float matC2B[16];
	if (!invertMatrix(matB2C, matC2B))
		return;

	for (int z = zstart; z <= zend; z++){
		for (int y = ystart; y <= yend; y++){
			for (int x = xstart; x <= xend; x++){
				if (x >= 0 && y >= 0 && z >= 0 &&
					x < volumeSize.width && y < volumeSize.height && z < volumeSize.depth)
				{
					float3 x1 = make_float3(x, y, z) * spacing -ave;
					float4 b = mat4mulvec4(matC2B, make_float4(x1, 1.0));

					if (b.x >= 0 && b.x <= 1
						&& b.y >= 0 && b.y <= 1
						&& b.z >= 0 && b.z <= 1
						&& b.w >= 0 && b.w <= 1){

						float3 x0 = b.x * vertsOri[0] + b.y * vertsOri[1] + b.z * vertsOri[2] + b.w * vertsOri[3];
						x0 = x0 / spacing;
						float res = tex3D(volumeTexInput, x0.x + 0.5, x0.y + 0.5, x0.z + 0.5);

						//float4 res = make_float4(0, 0, 0, 1.0*tetId / 450);
						surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
					}
				}
			}
		}
	}
}



bool PhysicalVolumeDeformProcessor::process(float* modelview, float* projection, int winWidth, int winHeight)
{
	if (!isActive)
		return false;

	if (meshDeformer->meshJustDeformed){
		float3 lensSpaceOrigin = meshDeformer->GetLensSpaceOrigin();
		float3 majorAxis = meshDeformer->GetXDiretion();
		float3 lensDir = meshDeformer->GetZDiretion();
		int3 nSteps = meshDeformer->GetNumSteps();
		float step = meshDeformer->GetStep();


		cudaExtent size = volume->volumeCuda.size;
		unsigned int dim = 32;
		dim3 blockSize(dim, dim, 1);
		dim3 gridSize(iDivUp(size.width, blockSize.x), iDivUp(size.height, blockSize.y), iDivUp(size.depth, blockSize.z));

		checkCudaErrors(cudaBindTextureToArray(volumeTexInput, volume->volumeCudaOri.content, volume->volumeCudaOri.channelDesc));

		checkCudaErrors(cudaBindSurfaceToArray(volumeSurfaceOut, volume->volumeCuda.content));

		float3 minorAxis = cross(lensDir, majorAxis);
		float3 range = make_float3(nSteps.x - 1, nSteps.y - 1, nSteps.z - 1)*step;
		d_updateVolumebyModelGrid_init << <gridSize, blockSize >> >(size, lensSpaceOrigin, majorAxis, minorAxis, lensDir, range, volume->spacing);


		dim3 blockSize2(8, 8, 8);
		dim3 gridSize2(meshDeformer->GetTetNumber(), 1, 1);
		d_updateVolumebyModelGrid << <gridSize2, blockSize2 >> >(size, meshDeformer->GetXDev(), meshDeformer->GetXDevOri(), meshDeformer->GetTetDev(), meshDeformer->GetTetNumber(), volume->spacing);

		checkCudaErrors(cudaUnbindTexture(volumeTexInput));
		meshDeformer->meshJustDeformed = false;
		return true;
	}
	else{
		return false;
	}
}


