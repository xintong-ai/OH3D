#include "ModelVolumeDeformer.h"
#include "TransformFunc.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

texture<float, 3, cudaReadModeElementType>  volumeTexInput;
surface<void, cudaSurfaceType3D>			volumeSurfaceOut;

inline int iDivUp22(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void ModelVolumeDeformer_KernelInit()
{
	volumeTexInput.normalized = false;
	volumeTexInput.filterMode = cudaFilterModeLinear;
	volumeTexInput.addressMode[0] = cudaAddressModeBorder;
	volumeTexInput.addressMode[1] = cudaAddressModeBorder;
	volumeTexInput.addressMode[2] = cudaAddressModeBorder;
}


void ModelVolumeDeformer::Init(Volume *_ori)
{
	originalVolume = _ori;

	volumeCUDADeformed.VolumeCUDA_init(_ori->size, _ori->values, 1, 1);

	volumeCUDAGradient.VolumeCUDA_init(_ori->size, 0, 1, 4);

	ModelVolumeDeformer_KernelInit();

	computeGradient();
}



__global__ void
d_updateVolumebyModelGrid_init(cudaExtent volumeSize)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
	{
		return;
	}
	float res = 0;
	surf3Dwrite(res, volumeSurfaceOut, x * sizeof(float), y, z);
	return;

}

__device__
inline bool within22(float v)
{
	return v >= 0 && v <= 1;
}


__device__
float Determinant4x422(const float4& v0,
	const float4& v1,
	const float4& v2,
	const float4& v3)
{
	float det = v0.w*v1.z*v2.y*v3.x - v0.z*v1.w*v2.y*v3.x -
		v0.w*v1.y*v2.z*v3.x + v0.y*v1.w*v2.z*v3.x +

		v0.z*v1.y*v2.w*v3.x - v0.y*v1.z*v2.w*v3.x -
		v0.w*v1.z*v2.x*v3.y + v0.z*v1.w*v2.x*v3.y +

		v0.w*v1.x*v2.z*v3.y - v0.x*v1.w*v2.z*v3.y -
		v0.z*v1.x*v2.w*v3.y + v0.x*v1.z*v2.w*v3.y +

		v0.w*v1.y*v2.x*v3.z - v0.y*v1.w*v2.x*v3.z -
		v0.w*v1.x*v2.y*v3.z + v0.x*v1.w*v2.y*v3.z +

		v0.y*v1.x*v2.w*v3.z - v0.x*v1.y*v2.w*v3.z -
		v0.z*v1.y*v2.x*v3.w + v0.y*v1.z*v2.x*v3.w +

		v0.z*v1.x*v2.y*v3.w - v0.x*v1.z*v2.y*v3.w -
		v0.y*v1.x*v2.z*v3.w + v0.x*v1.y*v2.z*v3.w;
	return det;
}

__device__
float4 GetBarycentricCoordinate22(const float3& v0_,
	const float3& v1_,
	const float3& v2_,
	const float3& v3_,
	const float3& p0_)
{
	float4 v0 = make_float4(v0_, 1);
	float4 v1 = make_float4(v1_, 1);
	float4 v2 = make_float4(v2_, 1);
	float4 v3 = make_float4(v3_, 1);
	float4 p0 = make_float4(p0_, 1);
	float4 barycentricCoord = float4();
	const float det0 = Determinant4x422(v0, v1, v2, v3);
	const float det1 = Determinant4x422(p0, v1, v2, v3);
	const float det2 = Determinant4x422(v0, p0, v2, v3);
	const float det3 = Determinant4x422(v0, v1, p0, v3);
	const float det4 = Determinant4x422(v0, v1, v2, p0);
	barycentricCoord.x = (det1 / det0);
	barycentricCoord.y = (det2 / det0);
	barycentricCoord.z = (det3 / det0);
	barycentricCoord.w = (det4 / det0);
	return barycentricCoord;
}

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
	//for (int z = ceil(volumeBoundmin.z) ; z <= floor(volumeBoundmax.z); z += 1){
	//	for (int y = ceil(volumeBoundmin.y) ; y <= floor(volumeBoundmax.y); y += 1){
	//		for (int x = ceil(volumeBoundmin.x) ; x <= floor(volumeBoundmax.x); x += 1){
	int zPieceSize = (floor(volumeBoundmax.z) - ceil(volumeBoundmin.z) + 1 + blockDim.z - 1) / blockDim.z;
	int zstart = ceil(volumeBoundmin.z) + zPieceSize*threadIdx.z, zend = min(zstart + zPieceSize - 1, (int)floor(volumeBoundmax.z));
	int yPieceSize = (floor(volumeBoundmax.y) - ceil(volumeBoundmin.y) + 1 + blockDim.y - 1) / blockDim.y;
	int ystart = ceil(volumeBoundmin.y) + yPieceSize*threadIdx.y, yend = min(ystart + yPieceSize - 1, (int)floor(volumeBoundmax.y));
	int xPieceSize = (floor(volumeBoundmax.x) - ceil(volumeBoundmin.x) + 1 + blockDim.x - 1) / blockDim.x;
	int xstart = ceil(volumeBoundmin.x) + xPieceSize*threadIdx.x, xend = min(xstart + xPieceSize - 1, (int)floor(volumeBoundmax.x));
	if (xend < 0 && yend < 0 && zend < 0 &&
		xstart >= volumeSize.width && ystart >= volumeSize.height && zstart >= volumeSize.depth){
		return;
	}

	for (int z = zstart; z <= zend; z++){
		for (int y = ystart; y <= yend; y++){
			for (int x = xstart; x <= xend; x++){
				//for (int z = ceil(volumeBoundmin.z) + threadIdx.z; z <= floor(volumeBoundmax.z); z += blockDim.z){
				//	for (int y = ceil(volumeBoundmin.y) + threadIdx.y; y <= floor(volumeBoundmax.y); y += blockDim.y){
				//		for (int x = ceil(volumeBoundmin.x) + threadIdx.x; x <= floor(volumeBoundmax.x); x += blockDim.x){
				if (x >= 0 && y >= 0 && z >= 0 &&
					x < volumeSize.width && y < volumeSize.height && z < volumeSize.depth)
				{
					float3 x1 = make_float3(x, y, z) * spacing;
					float4 b = GetBarycentricCoordinate22(verts[0], verts[1], verts[2], verts[3], x1);
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



void ModelVolumeDeformer::deformByModelGrid()
{
	cudaExtent size = volumeCUDADeformed.size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp22(size.width, blockSize.x), iDivUp22(size.height, blockSize.y), iDivUp22(size.depth, blockSize.z));

	checkCudaErrors(cudaBindSurfaceToArray(volumeSurfaceOut, volumeCUDADeformed.content));

	d_updateVolumebyModelGrid_init << <gridSize, blockSize >> >(size);

	checkCudaErrors(cudaBindTextureToArray(volumeTexInput, originalVolume->volumeCuda.content, originalVolume->volumeCuda.channelDesc));
	
	
	dim3 blockSize2(8, 8, 8);
	dim3 gridSize2(modelGrid->GetTetNumber(), 1, 1);
	d_updateVolumebyModelGrid << <gridSize2, blockSize2 >> >(size, modelGrid->GetXDev(), modelGrid->GetXDevOri(), modelGrid->GetTetDev(), modelGrid->GetTetNumber(), originalVolume->spacing);
	checkCudaErrors(cudaUnbindTexture(volumeTexInput));
}


__global__ void
d_computeGradient(cudaExtent volumeSize)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
	{
		return;
	}

	float4 grad = make_float4(0.0);

	int indz1 = z - 1, indz2 = z + 1;
	if (indz1 < 0)	indz1 = 0;
	if (indz2 > volumeSize.depth - 1) indz2 = volumeSize.depth - 1;
	grad.z = (tex3D(volumeTexInput, x + 0.5, y + 0.5, indz2 + 0.5) - tex3D(volumeTexInput, x + 0.5, y + 0.5, indz1 + 0.5)) / (indz2 - indz1);

	int indy1 = y - 1, indy2 = y + 1;
	if (indy1 < 0)	indy1 = 0;
	if (indy2 > y >= volumeSize.height - 1) indy2 = y >= volumeSize.height - 1;
	grad.y = (tex3D(volumeTexInput, x + 0.5, indy2 + 0.5, z + 0.5) - tex3D(volumeTexInput, x + 0.5, indy1 + 0.5, z + 0.5)) / (indy2 - indy1);

	int indx1 = x - 1, indx2 = x + 1;
	if (indx1 < 0)	indx1 = 0;
	if (indx2 > volumeSize.width - 1) indx2 = volumeSize.width - 1;
	grad.x = (tex3D(volumeTexInput, indx2 + 0.5, y + 0.5, z + 0.5) - tex3D(volumeTexInput, indx1 + 0.5, y + 0.5, z + 0.5)) / (indx2 - indx1);

	surf3Dwrite(grad, volumeSurfaceOut, x * sizeof(float4), y, z);
}

void ModelVolumeDeformer::computeGradient()
{
	cudaExtent size = volumeCUDADeformed.size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp22(size.width, blockSize.x), iDivUp22(size.height, blockSize.y), iDivUp22(size.depth, blockSize.z));

	checkCudaErrors(cudaBindTextureToArray(volumeTexInput, volumeCUDADeformed.content, volumeCUDADeformed.channelDesc));
	checkCudaErrors(cudaBindSurfaceToArray(volumeSurfaceOut, volumeCUDAGradient.content));

	d_computeGradient << <gridSize, blockSize >> >(size);
}

