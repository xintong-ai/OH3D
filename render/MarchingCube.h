#ifndef MRACHING_CUBE_H
#define MRACHING_CUBE_H

#include <memory>

#include <numeric>
#include <math.h>

#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>

#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_functions.h>
//#include <helper_cuda_gl.h>
#include <helper_math.h>

class Volume;

#include <Volume.h>
#include <PolyMesh.h>

typedef unsigned char uchar;

class MarchingCube 
{
public:
	std::shared_ptr<Volume> volume = 0;
	std::shared_ptr<PolyMesh> polyMesh = 0;

	float isoValue = 0.2f;
	bool needCompute;

	MarchingCube(std::shared_ptr<Volume> v, std::shared_ptr<PolyMesh> p, float value = 0.2)
	{
		volume = v;
		polyMesh = p;
		
		isoValue = 0.26;// value;

		initMC();

		computeIsosurface();
		//std::vector<float3> forDebug(maxVerts);
		//cudaMemcpy(&forDebug[0], d_pos, maxVerts*sizeof(float3), cudaMemcpyDeviceToHost);
		//int g = 0;
		//g += 9;
		//std::vector<float3> forDebug2(maxVerts);
		//cudaMemcpy(&forDebug2[0], d_normal, maxVerts*sizeof(float3), cudaMemcpyDeviceToHost);
		//g = 0;
		//g += 9;

		needCompute = false;
		updatePoly();
		polyMesh->find_center_and_range();  //just to compute the range, so one time call is enough
	}


	void updatePoly()
	{
		if (totalVerts > polyMesh->vertexcount || totalVerts / 3 > polyMesh->facecount)
		{
			polyMesh->~PolyMesh();

			polyMesh->vertexcount = totalVerts;
			polyMesh->facecount = totalVerts / 3;

			polyMesh->vertexCoords = new float[3 * polyMesh->vertexcount];
			polyMesh->vertexNorms = new float[3 * polyMesh->vertexcount];
			polyMesh->indices = new unsigned[3 * polyMesh->facecount];
		}
		else{
			polyMesh->vertexcount = totalVerts;
			polyMesh->facecount = totalVerts / 3;
		}

		cudaMemcpy(polyMesh->vertexCoords, d_pos, sizeof(float3)*polyMesh->vertexcount, cudaMemcpyDeviceToHost);
		cudaMemcpy(polyMesh->vertexNorms, d_normal, sizeof(float3)*polyMesh->vertexcount, cudaMemcpyDeviceToHost);
		std::iota(polyMesh->indices, polyMesh->indices + 3 * polyMesh->facecount, 0);

		////only needed when need to perform deformation
		//polyMesh->setVertexCoordsOri();
		//polyMesh->setVertexDeviateVals();
		//polyMesh->setVertexColorVals(0);
	}

private:
	void initMC();

	uint3 gridSizeLog2 = make_uint3(5, 5, 5);
	uint3 gridSizeShift;
	uint3 gridSize;
	uint3 gridSizeMask;

	float3 voxelSize;
	uint numVoxels = 0;
	uint maxVerts = 0;
	uint activeVoxels = 0;
	uint totalVerts = 0;

	uchar *d_volume = 0;
	uint *d_voxelVerts = 0;
	uint *d_voxelVertsScan = 0;
	uint *d_voxelOccupied = 0;
	uint *d_voxelOccupiedScan = 0;
	uint *d_compVoxelArray;

	//float4 *d_pos = 0, *d_normal = 0;
	float3 *d_pos = 0, *d_normal = 0;

	// tables
	uint *d_numVertsTable = 0;
	uint *d_edgeTable = 0;
	uint *d_triTable = 0;


	uchar *loadRawFile(char *filename, int size)
	{
		FILE *fp = fopen(filename, "rb");

		if (!fp)
		{
			fprintf(stderr, "Error opening file '%s'\n", filename);
			return 0;
		}

		uchar *data = (uchar *)malloc(size);
		size_t read = fread(data, 1, size, fp);
		fclose(fp);

		printf("Read '%s', %d bytes\n", filename, (int)read);

		return data;
	}


	unsigned char *getUCharVoxelValues(std::shared_ptr<Volume> v, uint3 gridSize){
		unsigned char * ret = new unsigned char[gridSize.x*gridSize.y*gridSize.z];
		memset(ret, 0, gridSize.x*gridSize.y*gridSize.z*sizeof(unsigned char));

		int3 size = v->size; // gridSize is larger or equal to size

/*		for (int k = 0; k < size.z; k++){
			for (int j = 0; j < size.y; j++){
				for (int i = 0; i < size.x; i++){	*/			
		for (int k = 0; k < min(size.z, gridSize.z); k++){
			for (int j = 0; j < min(size.y, gridSize.y); j++){
				for (int i = 0; i <min(size.x, gridSize.x); i++){
					ret[k*gridSize.y*gridSize.x + j*gridSize.x + i] = v->values[k*size.y*size.x + j*size.x + i] * 255;
				}
			}
		}

		return ret;
	}

	void bindVolumeTexture(uchar *d_volume);
	void allocateTextures(uint **d_edgeTable, uint **d_triTable, uint **d_numVertsTable);
	
	void computeIsosurface();

};

#endif //ARROW_RENDERABLE_H