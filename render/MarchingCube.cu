
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "MarchingCube.h"
#include "tables.h"


// if SAMPLE_VOLUME is 0, an implicit dataset is generated. If 1, a voxelized
// dataset is loaded from file
#define SAMPLE_VOLUME 1

// Using shared to store computed vertices and normals during triangle generation
// improves performance
#define USE_SHARED 1

// The number of threads to use for triangle generation (limited by shared memory size)
#define NTHREADS 32

#define SKIP_EMPTY_VOXELS 1


// volume data
texture<uchar, 1, cudaReadModeNormalizedFloat> volumeTex;
// textures containing look-up tables
texture<uint, 1, cudaReadModeElementType> edgeTex;
texture<uint, 1, cudaReadModeElementType> triTex;
texture<uint, 1, cudaReadModeElementType> numVertsTex;

void MarchingCube::bindVolumeTexture(uchar *d_volume)
{
	// bind to linear texture
	checkCudaErrors(cudaBindTexture(0, volumeTex, d_volume, cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned)));

}

void MarchingCube::allocateTextures(uint **d_edgeTable, uint **d_triTable, uint **d_numVertsTable)
{
	checkCudaErrors(cudaMalloc((void **)d_edgeTable, 256 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_edgeTable, (void *)edgeTable, 256 * sizeof(uint), cudaMemcpyHostToDevice));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	checkCudaErrors(cudaBindTexture(0, edgeTex, *d_edgeTable, channelDesc));

	checkCudaErrors(cudaMalloc((void **)d_triTable, 256 * 16 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_triTable, (void *)triTable, 256 * 16 * sizeof(uint), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, triTex, *d_triTable, channelDesc));

	checkCudaErrors(cudaMalloc((void **)d_numVertsTable, 256 * sizeof(uint)));
	checkCudaErrors(cudaMemcpy((void *)*d_numVertsTable, (void *)numVertsTable, 256 * sizeof(uint), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, numVertsTex, *d_numVertsTable, channelDesc));
}









void ThrustScanWrapper(unsigned int *output, unsigned int *input, unsigned int numElements)
{	
	thrust::exclusive_scan(thrust::device_ptr<unsigned int>(input),
		thrust::device_ptr<unsigned int>(input + numElements),
		thrust::device_ptr<unsigned int>(output));
}






// sample volume data set at a point
__device__
float sampleVolume(uchar *data, uint3 p, uint3 gridSize)
{
	p.x = min(p.x, gridSize.x - 1);
	p.y = min(p.y, gridSize.y - 1);
	p.z = min(p.z, gridSize.z - 1);
	uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
	//    return (float) data[i] / 255.0f;
	return tex1Dfetch(volumeTex, i);
}
// compute position in 3d grid from 1d index
// only works for power of 2 sizes
__device__
uint3 calcGridPos(uint i, uint3 gridSizeShift, uint3 gridSizeMask)
{
	uint3 gridPos;
	gridPos.x = i & gridSizeMask.x;
	gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
	gridPos.z = (i >> gridSizeShift.z) & gridSizeMask.z;
	return gridPos;
}
// classify voxel based on number of vertices it will generate
// one thread per voxel
__global__ void
classifyVoxel(uint *voxelVerts, uint *voxelOccupied, uchar *volume,
uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
float3 voxelSize, float isoValue)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	uint3 gridPos = calcGridPos(i, gridSizeShift, gridSizeMask);

	// read field values at neighbouring grid vertices
#if SAMPLE_VOLUME
	float field[8];
	field[0] = sampleVolume(volume, gridPos, gridSize);
	field[1] = sampleVolume(volume, gridPos + make_uint3(1, 0, 0), gridSize);
	field[2] = sampleVolume(volume, gridPos + make_uint3(1, 1, 0), gridSize);
	field[3] = sampleVolume(volume, gridPos + make_uint3(0, 1, 0), gridSize);
	field[4] = sampleVolume(volume, gridPos + make_uint3(0, 0, 1), gridSize);
	field[5] = sampleVolume(volume, gridPos + make_uint3(1, 0, 1), gridSize);
	field[6] = sampleVolume(volume, gridPos + make_uint3(1, 1, 1), gridSize);
	field[7] = sampleVolume(volume, gridPos + make_uint3(0, 1, 1), gridSize);
#else
	float3 p;
	p.x = -1.0f + (gridPos.x * voxelSize.x);
	p.y = -1.0f + (gridPos.y * voxelSize.y);
	p.z = -1.0f + (gridPos.z * voxelSize.z);

	float field[8];
	field[0] = fieldFunc(p);
	field[1] = fieldFunc(p + make_float3(voxelSize.x, 0, 0));
	field[2] = fieldFunc(p + make_float3(voxelSize.x, voxelSize.y, 0));
	field[3] = fieldFunc(p + make_float3(0, voxelSize.y, 0));
	field[4] = fieldFunc(p + make_float3(0, 0, voxelSize.z));
	field[5] = fieldFunc(p + make_float3(voxelSize.x, 0, voxelSize.z));
	field[6] = fieldFunc(p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z));
	field[7] = fieldFunc(p + make_float3(0, voxelSize.y, voxelSize.z));
#endif

	// calculate flag indicating if each vertex is inside or outside isosurface
	uint cubeindex;
	cubeindex = uint(field[0] < isoValue);
	cubeindex += uint(field[1] < isoValue) * 2;
	cubeindex += uint(field[2] < isoValue) * 4;
	cubeindex += uint(field[3] < isoValue) * 8;
	cubeindex += uint(field[4] < isoValue) * 16;
	cubeindex += uint(field[5] < isoValue) * 32;
	cubeindex += uint(field[6] < isoValue) * 64;
	cubeindex += uint(field[7] < isoValue) * 128;

	// read number of vertices from texture
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

	if (i < numVoxels)
	{
		voxelVerts[i] = numVerts;
		voxelOccupied[i] = (numVerts > 0);
	}
}

void
launch_classifyVoxel(dim3 grid, dim3 threads, uint *voxelVerts, uint *voxelOccupied, uchar *volume,
	uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask, uint numVoxels,
	float3 voxelSize, float isoValue)
{
	// calculate number of vertices need per voxel
	classifyVoxel << <grid, threads >> >(voxelVerts, voxelOccupied, volume,
		gridSize, gridSizeShift, gridSizeMask,
		numVoxels, voxelSize, isoValue);
	getLastCudaError("classifyVoxel failed");
}








// compact voxel array
__global__ void
compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (voxelOccupied[i] && (i < numVoxels))
	{
		compactedVoxelArray[voxelOccupiedScan[i]] = i;
	}
}

void
launch_compactVoxels(dim3 grid, dim3 threads, uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels)
{
	compactVoxels << <grid, threads >> >(compactedVoxelArray, voxelOccupied,
		voxelOccupiedScan, numVoxels);
	getLastCudaError("compactVoxels failed");
}





// compute interpolated vertex along an edge
__device__
float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
	float t = (isolevel - f0) / (f1 - f0);
	return lerp(p0, p1, t);
}

// calculate triangle normal
__device__
float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{
	float3 edge0 = *v1 - *v0;
	float3 edge1 = *v2 - *v0;
	// note - it's faster to perform normalization in vertex shader rather than here
	return cross(edge0, edge1);
}

// version that calculates flat surface normal for each triangle
__global__ void
//generateTriangles2(float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, uchar *volume,
generateTriangles2(float3 *pos, float3 *norm, uint *compactedVoxelArray, uint *numVertsScanned, uchar *volume,
uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
	uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

	if (i > activeVoxels - 1)
	{
		i = activeVoxels - 1;
	}

#if SKIP_EMPTY_VOXELS
	uint voxel = compactedVoxelArray[i];
#else
	uint voxel = i;
#endif

	// compute position in 3d grid
	uint3 gridPos = calcGridPos(voxel, gridSizeShift, gridSizeMask);

	float3 p;
	p.x = -1.0f + (gridPos.x * voxelSize.x);
	p.y = -1.0f + (gridPos.y * voxelSize.y);
	p.z = -1.0f + (gridPos.z * voxelSize.z);

	// calculate cell vertex positions
	float3 v[8];
	v[0] = p;
	v[1] = p + make_float3(voxelSize.x, 0, 0);
	v[2] = p + make_float3(voxelSize.x, voxelSize.y, 0);
	v[3] = p + make_float3(0, voxelSize.y, 0);
	v[4] = p + make_float3(0, 0, voxelSize.z);
	v[5] = p + make_float3(voxelSize.x, 0, voxelSize.z);
	v[6] = p + make_float3(voxelSize.x, voxelSize.y, voxelSize.z);
	v[7] = p + make_float3(0, voxelSize.y, voxelSize.z);

#if SAMPLE_VOLUME
	float field[8];
	field[0] = sampleVolume(volume, gridPos, gridSize);
	field[1] = sampleVolume(volume, gridPos + make_uint3(1, 0, 0), gridSize);
	field[2] = sampleVolume(volume, gridPos + make_uint3(1, 1, 0), gridSize);
	field[3] = sampleVolume(volume, gridPos + make_uint3(0, 1, 0), gridSize);
	field[4] = sampleVolume(volume, gridPos + make_uint3(0, 0, 1), gridSize);
	field[5] = sampleVolume(volume, gridPos + make_uint3(1, 0, 1), gridSize);
	field[6] = sampleVolume(volume, gridPos + make_uint3(1, 1, 1), gridSize);
	field[7] = sampleVolume(volume, gridPos + make_uint3(0, 1, 1), gridSize);
#else
	// evaluate field values
	float field[8];
	field[0] = fieldFunc(v[0]);
	field[1] = fieldFunc(v[1]);
	field[2] = fieldFunc(v[2]);
	field[3] = fieldFunc(v[3]);
	field[4] = fieldFunc(v[4]);
	field[5] = fieldFunc(v[5]);
	field[6] = fieldFunc(v[6]);
	field[7] = fieldFunc(v[7]);
#endif

	// recalculate flag
	uint cubeindex;
	cubeindex = uint(field[0] < isoValue);
	cubeindex += uint(field[1] < isoValue) * 2;
	cubeindex += uint(field[2] < isoValue) * 4;
	cubeindex += uint(field[3] < isoValue) * 8;
	cubeindex += uint(field[4] < isoValue) * 16;
	cubeindex += uint(field[5] < isoValue) * 32;
	cubeindex += uint(field[6] < isoValue) * 64;
	cubeindex += uint(field[7] < isoValue) * 128;

	// find the vertices where the surface intersects the cube

#if USE_SHARED
	// use shared memory to avoid using local
	__shared__ float3 vertlist[12 * NTHREADS];

	vertlist[threadIdx.x] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
	vertlist[NTHREADS + threadIdx.x] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
	vertlist[(NTHREADS * 2) + threadIdx.x] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
	vertlist[(NTHREADS * 3) + threadIdx.x] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);
	vertlist[(NTHREADS * 4) + threadIdx.x] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
	vertlist[(NTHREADS * 5) + threadIdx.x] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
	vertlist[(NTHREADS * 6) + threadIdx.x] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
	vertlist[(NTHREADS * 7) + threadIdx.x] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);
	vertlist[(NTHREADS * 8) + threadIdx.x] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
	vertlist[(NTHREADS * 9) + threadIdx.x] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
	vertlist[(NTHREADS * 10) + threadIdx.x] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
	vertlist[(NTHREADS * 11) + threadIdx.x] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
	__syncthreads();
#else

	float3 vertlist[12];

	vertlist[0] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
	vertlist[1] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
	vertlist[2] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
	vertlist[3] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);

	vertlist[4] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
	vertlist[5] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
	vertlist[6] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
	vertlist[7] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);

	vertlist[8] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
	vertlist[9] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
	vertlist[10] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
	vertlist[11] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
#endif

	// output triangle vertices
	uint numVerts = tex1Dfetch(numVertsTex, cubeindex);

	for (int i = 0; i<numVerts; i += 3)
	{
		uint index = numVertsScanned[voxel] + i;

		float3 *v[3];
		uint edge;
		edge = tex1Dfetch(triTex, (cubeindex * 16) + i);
#if USE_SHARED
		v[0] = &vertlist[(edge*NTHREADS) + threadIdx.x];
#else
		v[0] = &vertlist[edge];
#endif

		edge = tex1Dfetch(triTex, (cubeindex * 16) + i + 1);
#if USE_SHARED
		v[1] = &vertlist[(edge*NTHREADS) + threadIdx.x];
#else
		v[1] = &vertlist[edge];
#endif

		edge = tex1Dfetch(triTex, (cubeindex * 16) + i + 2);
#if USE_SHARED
		v[2] = &vertlist[(edge*NTHREADS) + threadIdx.x];
#else
		v[2] = &vertlist[edge];
#endif

		// calculate triangle surface normal
		float3 n = calcNormal(v[0], v[1], v[2]);

		if (index < (maxVerts - 3))
		{
			//pos[index] = make_float4(*v[0], 1.0f);
			//norm[index] = make_float4(n, 0.0f);

			//pos[index + 1] = make_float4(*v[1], 1.0f);
			//norm[index + 1] = make_float4(n, 0.0f);

			//pos[index + 2] = make_float4(*v[2], 1.0f);
			//norm[index + 2] = make_float4(n, 0.0f);



			//Added by CL !!!!!!!! only for float3
			pos[index] = *v[0];
			norm[index] = n;

			pos[index + 1] = *v[1];
			norm[index + 1] = n;

			pos[index + 2] = *v[2];
			norm[index + 2] = n;

			pos[index] = (pos[index] + 1.0) / 2.0 * gridSize.x;
			pos[index + 1] = (pos[index + 1] + 1.0) / 2.0 * gridSize.y;
			pos[index + 2] = (pos[index + 2] + 1.0) / 2.0 * gridSize.z;

		}
	}
}



void
//launch_generateTriangles2(dim3 grid, dim3 threads,
//float4 *pos, float4 *norm, uint *compactedVoxelArray, uint *numVertsScanned, uchar *volume,
//uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
//float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts)
launch_generateTriangles2(dim3 grid, dim3 threads,
float3 *pos, float3 *norm, uint *compactedVoxelArray, uint *numVertsScanned, uchar *volume,
uint3 gridSize, uint3 gridSizeShift, uint3 gridSizeMask,
float3 voxelSize, float isoValue, uint activeVoxels, uint maxVerts)
{
	generateTriangles2 << <grid, NTHREADS >> >(pos, norm,
		compactedVoxelArray,
		numVertsScanned, volume,
		gridSize, gridSizeShift, gridSizeMask,
		voxelSize, isoValue, activeVoxels,
		maxVerts);
	getLastCudaError("generateTriangles2 failed");
}

void MarchingCube::computeIsosurface()
{
	int threads = 128;
	dim3 grid(numVoxels / threads, 1, 1);

	// get around maximum grid size of 65535 in each dimension
	if (grid.x > 65535)
	{
		grid.y = grid.x / 32768;
		grid.x = 32768;
	}

	// calculate number of vertices need per voxel
	launch_classifyVoxel(grid, threads,
		d_voxelVerts, d_voxelOccupied, d_volume,
		gridSize, gridSizeShift, gridSizeMask,
		numVoxels, voxelSize, isoValue);

#if DEBUG_BUFFERS
	printf("voxelVerts:\n");
	dumpBuffer(d_voxelVerts, numVoxels, sizeof(uint));
#endif

#if SKIP_EMPTY_VOXELS
	// scan voxel occupied array
	ThrustScanWrapper(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

#if DEBUG_BUFFERS
	printf("voxelOccupiedScan:\n");
	dumpBuffer(d_voxelOccupiedScan, numVoxels, sizeof(uint));
#endif


	// read back values to calculate total number of non-empty voxels
	// since we are using an exclusive scan, the total is the last value of
	// the scan result plus the last value in the input array
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
			(void *)(d_voxelOccupied + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
			(void *)(d_voxelOccupiedScan + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		activeVoxels = lastElement + lastScanElement;
	}

	if (activeVoxels == 0)
	{
		// return if there are no full voxels
		totalVerts = 0;
		return;
	}

	// compact voxel index array
	launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);
	getLastCudaError("compactVoxels failed");

#endif // SKIP_EMPTY_VOXELS


	// scan voxel vertex count array
	ThrustScanWrapper(d_voxelVertsScan, d_voxelVerts, numVoxels);

#if DEBUG_BUFFERS
	printf("voxelVertsScan:\n");
	dumpBuffer(d_voxelVertsScan, numVoxels, sizeof(uint));
#endif

	// readback total number of vertices
	{
		uint lastElement, lastScanElement;
		checkCudaErrors(cudaMemcpy((void *)&lastElement,
			(void *)(d_voxelVerts + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy((void *)&lastScanElement,
			(void *)(d_voxelVertsScan + numVoxels - 1),
			sizeof(uint), cudaMemcpyDeviceToHost));
		totalVerts = lastElement + lastScanElement;
	}

#if SKIP_EMPTY_VOXELS
	dim3 grid2((int)ceil(activeVoxels / (float)NTHREADS), 1, 1);
#else
	dim3 grid2((int)ceil(numVoxels / (float)NTHREADS), 1, 1);
#endif

	while (grid2.x > 65535)
	{
		grid2.x /= 2;
		grid2.y *= 2;
	}

#if SAMPLE_VOLUME
	launch_generateTriangles2(grid2, NTHREADS, d_pos, d_normal,
		d_compVoxelArray,
		d_voxelVertsScan, d_volume,
		gridSize, gridSizeShift, gridSizeMask,
		voxelSize, isoValue, activeVoxels,
		maxVerts);
#else
	launch_generateTriangles(grid2, NTHREADS, d_pos, d_normal,
		d_compVoxelArray,
		d_voxelVertsScan,
		gridSize, gridSizeShift, gridSizeMask,
		voxelSize, isoValue, activeVoxels,
		maxVerts);
#endif
}




void MarchingCube::initMC()
{
	//gridSizeLog2 = make_uint3(ceil(log2(volume->size.x)), ceil(log2(volume->size.y)), ceil(log2(volume->size.z)));
	gridSizeLog2 = make_uint3(7, 7, 7);
	//std::cout << "gridSizeLog2: " << gridSizeLog2.x << " " << gridSizeLog2.y << " " << gridSizeLog2.z << std::endl;
	gridSize = make_uint3(1 << gridSizeLog2.x, 1 << gridSizeLog2.y, 1 << gridSizeLog2.z);
	gridSizeMask = make_uint3(gridSize.x - 1, gridSize.y - 1, gridSize.z - 1);
	gridSizeShift = make_uint3(0, gridSizeLog2.x, gridSizeLog2.x + gridSizeLog2.y);

	numVoxels = gridSize.x*gridSize.y*gridSize.z;
	voxelSize = make_float3(2.0f / gridSize.x, 2.0f / gridSize.y, 2.0f / gridSize.z);
	maxVerts = gridSize.x*gridSize.y * 100;

	printf("grid: %d x %d x %d = %d voxels\n", gridSize.x, gridSize.y, gridSize.z, numVoxels);
	printf("max verts = %d\n", maxVerts);


	int size = gridSize.x*gridSize.y*gridSize.z*sizeof(unsigned char);
	unsigned char *volumearray = getUCharVoxelValues(volume, gridSize);

	//FILE *fp = fopen("test.raw", "wb");
	//fwrite(volumearray, 1, gridSize.x*gridSize.y*gridSize.z, fp);
	//fclose(fp);

	checkCudaErrors(cudaMalloc((void **)&d_volume, size));
	checkCudaErrors(cudaMemcpy(d_volume, volumearray, size, cudaMemcpyHostToDevice));
	free(volumearray);


	//int size = gridSize.x*gridSize.y*gridSize.z*sizeof(uchar);
	//uchar *volume = loadRawFile("Bucky.raw", size);
	//checkCudaErrors(cudaMalloc((void **)&d_volume, size));
	//checkCudaErrors(cudaMemcpy(d_volume, volume, size, cudaMemcpyHostToDevice));
	//free(volume);

	bindVolumeTexture(d_volume);


	//cudaMalloc((void **)&(d_pos), maxVerts*sizeof(float)* 4);
	//cudaMalloc((void **)&(d_normal), maxVerts*sizeof(float)* 4);
	cudaMalloc((void **)&(d_pos), maxVerts*sizeof(float)* 3);
	cudaMalloc((void **)&(d_normal), maxVerts*sizeof(float)* 3);

	// allocate textures
	allocateTextures(&d_edgeTable, &d_triTable, &d_numVertsTable);

	// allocate device memory
	unsigned int memSize = sizeof(uint)* numVoxels;
	checkCudaErrors(cudaMalloc((void **)&d_voxelVerts, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_voxelVertsScan, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_voxelOccupied, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_voxelOccupiedScan, memSize));
	checkCudaErrors(cudaMalloc((void **)&d_compVoxelArray, memSize));
}
