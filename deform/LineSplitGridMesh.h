///////////////////////////////////////////////////////////////////////////////////////////
//  Class LineSplitGridMesh
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef LINESPLIT_GRID_MESH_H
#define LINESPLIT_GRID_MESH_H
#include "physics/CUDA_PROJECTIVE_TET_MESH.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

__global__ void d_moveMeshNodes(float* X, float* X_Orig, int number, float3 moveDir)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= number)	return;
	int idx = i*3;
	X[idx + 0] = X[idx + 0] + moveDir.x;
	X[idx + 1] = X[idx + 1] + moveDir.y;
	X[idx + 2] = X[idx + 2] + moveDir.z;
	X_Orig[idx + 0] = X_Orig[idx + 0] + moveDir.x;
	X_Orig[idx + 1] = X_Orig[idx + 1] + moveDir.y;
	X_Orig[idx + 2] = X_Orig[idx + 2] + moveDir.z;
}

template <class TYPE>
class LineSplitGridMesh : public CUDA_PROJECTIVE_TET_MESH<TYPE>
{

	inline int IdxConv(int idx, int nStep[3], int3 vc)
	{
		return idx + vc.x * nStep[1] * nStep[2] + vc.y * nStep[2] + vc.z;
	}

	inline int IdxConv(int i, int j, int k, int nStep[3], int3 vc)
	{
		return (i + vc.x) * nStep[1] * nStep[2] + (j + vc.y) * nStep[2] + k + vc.z;
	}

	int IdxConvForSplittingNodes(int idx, int nStep[3], int3 vc)
	{
		if (vc.y == 0){
			int originali = idx / (nStep[1] * nStep[2]);
			int originalj = (idx - originali*nStep[1] * nStep[2]) / nStep[2];
			int originalk = idx - originali*nStep[1] * nStep[2] - originalj* nStep[2];
			int regularAllIdx = nStep[0] * nStep[1] * nStep[2];

			if (originali == 0){
				if (vc.x == 0){
					return  IdxConv(idx, nStep, vc);
				}
				else{
					return regularAllIdx + (originalk + vc.z)*(nStep[0] - 2);
				}
			}
			else if (originali <nStep[0]-2){
				return regularAllIdx + (originalk + vc.z)*(nStep[0] - 2) + originali - 1 + vc.x;
			}
			else{
				if (vc.x == 1){
					return  IdxConv(idx, nStep, vc);
				}
				else{
					return regularAllIdx + (originalk + vc.z)*(nStep[0] - 2) + nStep[0] - 3;
				}
			}
		}
		else{
			return  IdxConv(idx, nStep, vc);
		}
	}

	template <class TYPE>
	inline void Copy3(TYPE from[3], TYPE to[3])
	{
		to[0] = from[0];
		to[1] = from[1];
		to[2] = from[2];
	}


public:
	//using TET_MESH<TYPE>::tet_number;
	//using TET_MESH<TYPE>::number;
	//using TET_MESH<TYPE>::Tet;
	//using TET_MESH<TYPE>::inv_Dm;
	//using TET_MESH<TYPE>::Vol;
	//using TET_MESH<TYPE>::X;
	//using TET_MESH<TYPE>::Dm;
	//using CUDA_PROJECTIVE_TET_MESH<TYPE>::control_mag;
	//using TET_MESH<TYPE>::max_number;
	//using TET_MESH<TYPE>::M;
	//using TET_MESH<TYPE>::t_number;
	//using TET_MESH<TYPE>::T;
	//using TET_MESH<TYPE>::VN;
	//using TET_MESH<TYPE>::TN;
	//using TET_MESH<TYPE>::l_number;
	//using TET_MESH<TYPE>::L;
	//using CUDA_PROJECTIVE_TET_MESH<TYPE>::damping;


	int nStep[3];
	float step;

	int cutY;

	float3 lensSpaceOriginInWorld;




	//used for temperary mesh
	LineSplitGridMesh() : CUDA_PROJECTIVE_TET_MESH<TYPE>()
	{
		number = 0;
		tet_number = 0;
		return;
	}

	~LineSplitGridMesh(){};


	LineSplitGridMesh(float ztop, float zbottom, int meshRes, float3 lensCenter, float lSemiMajorAxis, float lSemiWidth, float3 majorAxis, float3 lensDir, glm::mat4 &meshTransMat) : CUDA_PROJECTIVE_TET_MESH<TYPE>()
	{
		//computeShapeInfo and computeInitCoord define a mesh that covers the lens region and nearby region
		computeShapeInfo(ztop, zbottom, meshRes, lensCenter, lSemiMajorAxis, lSemiWidth, majorAxis, lensDir);

		initLocalMem_CUDA_PROJECTIVE_TET_MESH();

		computeInitCoord(majorAxis, lensDir, meshTransMat);

		BuildTet();
		Build_Boundary_Lines();


		//printf("N: %d, %d\n", number, tet_number);

		control_mag = 500;		//500
		damping = 0.9;

		Allocate_GPU_Memory_InAdvance();

		return;

	}

	void Build_Boundary_Lines(){
		l_number = tet_number * 6;
		for (int i = 0; i < tet_number; i++) {
			L[i * 6 * 2 + 0] = Tet[i * 4 + 0];
			L[i * 6 * 2 + 1] = Tet[i * 4 + 1];
			L[i * 6 * 2 + 2] = Tet[i * 4 + 0];
			L[i * 6 * 2 + 3] = Tet[i * 4 + 2];
			L[i * 6 * 2 + 4] = Tet[i * 4 + 0];
			L[i * 6 * 2 + 5] = Tet[i * 4 + 3];
			L[i * 6 * 2 + 6] = Tet[i * 4 + 1];
			L[i * 6 * 2 + 7] = Tet[i * 4 + 2];
			L[i * 6 * 2 + 8] = Tet[i * 4 + 2];
			L[i * 6 * 2 + 9] = Tet[i * 4 + 3];
			L[i * 6 * 2 + 10] = Tet[i * 4 + 3];
			L[i * 6 * 2 + 11] = Tet[i * 4 + 1];
		}
	}

	void PrintMesh(){
		for (int i = 0; i < nStep[0]; i++){
			for (int j = 0; j < nStep[1]; j++){
				for (int k = 0; k < nStep[2]; k++){
					int idx = i * nStep[1] * nStep[2] + j * nStep[2] + k;
					std::cout << X[3 * idx + 0] << " "
						<< X[3 * idx + 1] << " "
						<< X[3 * idx + 2] << ";" << std::endl;
				}
			}
		}
		for (int i = 0; i < (nStep[0] - 1); i++){
			for (int j = 0; j < (nStep[1] - 1); j++){
				for (int k = 0; k < (nStep[2] - 1); k++){
					int idx = i * (nStep[1] - 1) * (nStep[2] - 1) + j * (nStep[2] - 1) + k;
					for (int t = 0; t < 5; t++){
						for (int v = 0; v < 4; v++){
							std::cout << Tet[4 * 5 * idx + 4 * t + v] << " ";
						}
						std::cout << ";" << std::endl;
					}
				}
			}
		}
	}

	void BuildTet()
	{
		
		tetVolumeOriginal = new float[tet_number];
		float tetVolumeCandidate1 = step*step*step / 6, tetVolumeCandidate2 = step*step*step / 3;
		
		int3 vc[8];
		vc[0] = make_int3(0, 0, 0);
		vc[1] = make_int3(1, 0, 0);
		vc[2] = make_int3(0, 1, 0);
		vc[3] = make_int3(1, 1, 0);
		vc[4] = make_int3(0, 0, 1);
		vc[5] = make_int3(1, 0, 1);
		vc[6] = make_int3(0, 1, 1);
		vc[7] = make_int3(1, 1, 1);

		int idx, idx2;
		for (int i = 0; i < (nStep[0] - 1); i++){
			for (int j = 0; j < (nStep[1] - 1); j++){
				for (int k = 0; k < (nStep[2] - 1); k++){
					if (j == cutY){
						idx = i * (nStep[1] - 1) * (nStep[2] - 1) + j * (nStep[2] - 1) + k;
						idx2 = i * nStep[1] * nStep[2] + j * nStep[2] + k;
						
						tetVolumeOriginal[idx * 5 + 0] = tetVolumeCandidate1;
						tetVolumeOriginal[idx * 5 + 1] = tetVolumeCandidate1;
						tetVolumeOriginal[idx * 5 + 2] = tetVolumeCandidate1;
						tetVolumeOriginal[idx * 5 + 3] = tetVolumeCandidate1;
						tetVolumeOriginal[idx * 5 + 4] = tetVolumeCandidate2;

						if ((i + j + k) % 2 == 0) {
							Tet[idx * 5 * 4 + 4 * 0 + 0] = IdxConvForSplittingNodes(idx2, nStep, vc[0]);
							Tet[idx * 5 * 4 + 4 * 0 + 1] = IdxConvForSplittingNodes(idx2, nStep, vc[1]);
							Tet[idx * 5 * 4 + 4 * 0 + 2] = IdxConvForSplittingNodes(idx2, nStep, vc[2]);
							Tet[idx * 5 * 4 + 4 * 0 + 3] = IdxConvForSplittingNodes(idx2, nStep, vc[4]);

							Tet[idx * 5 * 4 + 4 * 1 + 0] = IdxConvForSplittingNodes(idx2, nStep, vc[3]);
							Tet[idx * 5 * 4 + 4 * 1 + 1] = IdxConvForSplittingNodes(idx2, nStep, vc[1]);
							Tet[idx * 5 * 4 + 4 * 1 + 2] = IdxConvForSplittingNodes(idx2, nStep, vc[2]);
							Tet[idx * 5 * 4 + 4 * 1 + 3] = IdxConvForSplittingNodes(idx2, nStep, vc[7]);

							Tet[idx * 5 * 4 + 4 * 2 + 0] = IdxConvForSplittingNodes(idx2, nStep, vc[4]);
							Tet[idx * 5 * 4 + 4 * 2 + 1] = IdxConvForSplittingNodes(idx2, nStep, vc[5]);
							Tet[idx * 5 * 4 + 4 * 2 + 2] = IdxConvForSplittingNodes(idx2, nStep, vc[1]);
							Tet[idx * 5 * 4 + 4 * 2 + 3] = IdxConvForSplittingNodes(idx2, nStep, vc[7]);

							Tet[idx * 5 * 4 + 4 * 3 + 0] = IdxConvForSplittingNodes(idx2, nStep, vc[2]);
							Tet[idx * 5 * 4 + 4 * 3 + 1] = IdxConvForSplittingNodes(idx2, nStep, vc[4]);
							Tet[idx * 5 * 4 + 4 * 3 + 2] = IdxConvForSplittingNodes(idx2, nStep, vc[6]);
							Tet[idx * 5 * 4 + 4 * 3 + 3] = IdxConvForSplittingNodes(idx2, nStep, vc[7]);

							Tet[idx * 5 * 4 + 4 * 4 + 0] = IdxConvForSplittingNodes(idx2, nStep, vc[1]);
							Tet[idx * 5 * 4 + 4 * 4 + 1] = IdxConvForSplittingNodes(idx2, nStep, vc[2]);
							Tet[idx * 5 * 4 + 4 * 4 + 2] = IdxConvForSplittingNodes(idx2, nStep, vc[4]);
							Tet[idx * 5 * 4 + 4 * 4 + 3] = IdxConvForSplittingNodes(idx2, nStep, vc[7]);
						}
						else{
							Tet[idx * 5 * 4 + 4 * 0 + 0] = IdxConvForSplittingNodes(idx2, nStep, vc[0]);
							Tet[idx * 5 * 4 + 4 * 0 + 1] = IdxConvForSplittingNodes(idx2, nStep, vc[1]);
							Tet[idx * 5 * 4 + 4 * 0 + 2] = IdxConvForSplittingNodes(idx2, nStep, vc[3]);
							Tet[idx * 5 * 4 + 4 * 0 + 3] = IdxConvForSplittingNodes(idx2, nStep, vc[5]);

							Tet[idx * 5 * 4 + 4 * 1 + 0] = IdxConvForSplittingNodes(idx2, nStep, vc[0]);
							Tet[idx * 5 * 4 + 4 * 1 + 1] = IdxConvForSplittingNodes(idx2, nStep, vc[2]);
							Tet[idx * 5 * 4 + 4 * 1 + 2] = IdxConvForSplittingNodes(idx2, nStep, vc[3]);
							Tet[idx * 5 * 4 + 4 * 1 + 3] = IdxConvForSplittingNodes(idx2, nStep, vc[6]);

							Tet[idx * 5 * 4 + 4 * 2 + 0] = IdxConvForSplittingNodes(idx2, nStep, vc[0]);
							Tet[idx * 5 * 4 + 4 * 2 + 1] = IdxConvForSplittingNodes(idx2, nStep, vc[4]);
							Tet[idx * 5 * 4 + 4 * 2 + 2] = IdxConvForSplittingNodes(idx2, nStep, vc[5]);
							Tet[idx * 5 * 4 + 4 * 2 + 3] = IdxConvForSplittingNodes(idx2, nStep, vc[6]);

							Tet[idx * 5 * 4 + 4 * 3 + 0] = IdxConvForSplittingNodes(idx2, nStep, vc[3]);
							Tet[idx * 5 * 4 + 4 * 3 + 1] = IdxConvForSplittingNodes(idx2, nStep, vc[5]);
							Tet[idx * 5 * 4 + 4 * 3 + 2] = IdxConvForSplittingNodes(idx2, nStep, vc[6]);
							Tet[idx * 5 * 4 + 4 * 3 + 3] = IdxConvForSplittingNodes(idx2, nStep, vc[7]);

							Tet[idx * 5 * 4 + 4 * 4 + 0] = IdxConvForSplittingNodes(idx2, nStep, vc[0]);
							Tet[idx * 5 * 4 + 4 * 4 + 1] = IdxConvForSplittingNodes(idx2, nStep, vc[3]);
							Tet[idx * 5 * 4 + 4 * 4 + 2] = IdxConvForSplittingNodes(idx2, nStep, vc[5]);
							Tet[idx * 5 * 4 + 4 * 4 + 3] = IdxConvForSplittingNodes(idx2, nStep, vc[6]);
						}
					}
					else{
						idx = i * (nStep[1] - 1) * (nStep[2] - 1) + j * (nStep[2] - 1) + k;
						idx2 = i * nStep[1] * nStep[2] + j * nStep[2] + k;
						
						tetVolumeOriginal[idx * 5 + 0] = tetVolumeCandidate1;
						tetVolumeOriginal[idx * 5 + 1] = tetVolumeCandidate1;
						tetVolumeOriginal[idx * 5 + 2] = tetVolumeCandidate1;
						tetVolumeOriginal[idx * 5 + 3] = tetVolumeCandidate1;
						tetVolumeOriginal[idx * 5 + 4] = tetVolumeCandidate2; 
						
						if ((i + j + k) % 2 == 0) {
							Tet[idx * 5 * 4 + 4 * 0 + 0] = IdxConv(idx2, nStep, vc[0]);
							Tet[idx * 5 * 4 + 4 * 0 + 1] = IdxConv(idx2, nStep, vc[1]);
							Tet[idx * 5 * 4 + 4 * 0 + 2] = IdxConv(idx2, nStep, vc[2]);
							Tet[idx * 5 * 4 + 4 * 0 + 3] = IdxConv(idx2, nStep, vc[4]);

							Tet[idx * 5 * 4 + 4 * 1 + 0] = IdxConv(idx2, nStep, vc[3]);
							Tet[idx * 5 * 4 + 4 * 1 + 1] = IdxConv(idx2, nStep, vc[1]);
							Tet[idx * 5 * 4 + 4 * 1 + 2] = IdxConv(idx2, nStep, vc[2]);
							Tet[idx * 5 * 4 + 4 * 1 + 3] = IdxConv(idx2, nStep, vc[7]);

							Tet[idx * 5 * 4 + 4 * 2 + 0] = IdxConv(idx2, nStep, vc[4]);
							Tet[idx * 5 * 4 + 4 * 2 + 1] = IdxConv(idx2, nStep, vc[5]);
							Tet[idx * 5 * 4 + 4 * 2 + 2] = IdxConv(idx2, nStep, vc[1]);
							Tet[idx * 5 * 4 + 4 * 2 + 3] = IdxConv(idx2, nStep, vc[7]);

							Tet[idx * 5 * 4 + 4 * 3 + 0] = IdxConv(idx2, nStep, vc[2]);
							Tet[idx * 5 * 4 + 4 * 3 + 1] = IdxConv(idx2, nStep, vc[4]);
							Tet[idx * 5 * 4 + 4 * 3 + 2] = IdxConv(idx2, nStep, vc[6]);
							Tet[idx * 5 * 4 + 4 * 3 + 3] = IdxConv(idx2, nStep, vc[7]);

							Tet[idx * 5 * 4 + 4 * 4 + 0] = IdxConv(idx2, nStep, vc[1]);
							Tet[idx * 5 * 4 + 4 * 4 + 1] = IdxConv(idx2, nStep, vc[2]);
							Tet[idx * 5 * 4 + 4 * 4 + 2] = IdxConv(idx2, nStep, vc[4]);
							Tet[idx * 5 * 4 + 4 * 4 + 3] = IdxConv(idx2, nStep, vc[7]);
						}
						else{
							Tet[idx * 5 * 4 + 4 * 0 + 0] = IdxConv(idx2, nStep, vc[0]);
							Tet[idx * 5 * 4 + 4 * 0 + 1] = IdxConv(idx2, nStep, vc[1]);
							Tet[idx * 5 * 4 + 4 * 0 + 2] = IdxConv(idx2, nStep, vc[3]);
							Tet[idx * 5 * 4 + 4 * 0 + 3] = IdxConv(idx2, nStep, vc[5]);

							Tet[idx * 5 * 4 + 4 * 1 + 0] = IdxConv(idx2, nStep, vc[0]);
							Tet[idx * 5 * 4 + 4 * 1 + 1] = IdxConv(idx2, nStep, vc[2]);
							Tet[idx * 5 * 4 + 4 * 1 + 2] = IdxConv(idx2, nStep, vc[3]);
							Tet[idx * 5 * 4 + 4 * 1 + 3] = IdxConv(idx2, nStep, vc[6]);

							Tet[idx * 5 * 4 + 4 * 2 + 0] = IdxConv(idx2, nStep, vc[0]);
							Tet[idx * 5 * 4 + 4 * 2 + 1] = IdxConv(idx2, nStep, vc[4]);
							Tet[idx * 5 * 4 + 4 * 2 + 2] = IdxConv(idx2, nStep, vc[5]);
							Tet[idx * 5 * 4 + 4 * 2 + 3] = IdxConv(idx2, nStep, vc[6]);

							Tet[idx * 5 * 4 + 4 * 3 + 0] = IdxConv(idx2, nStep, vc[3]);
							Tet[idx * 5 * 4 + 4 * 3 + 1] = IdxConv(idx2, nStep, vc[5]);
							Tet[idx * 5 * 4 + 4 * 3 + 2] = IdxConv(idx2, nStep, vc[6]);
							Tet[idx * 5 * 4 + 4 * 3 + 3] = IdxConv(idx2, nStep, vc[7]);

							Tet[idx * 5 * 4 + 4 * 4 + 0] = IdxConv(idx2, nStep, vc[0]);
							Tet[idx * 5 * 4 + 4 * 4 + 1] = IdxConv(idx2, nStep, vc[3]);
							Tet[idx * 5 * 4 + 4 * 4 + 2] = IdxConv(idx2, nStep, vc[5]);
							Tet[idx * 5 * 4 + 4 * 4 + 3] = IdxConv(idx2, nStep, vc[6]);

						}
					}
				}
			}
		}
		cudaMalloc((void**)&dev_tetVolumeOriginal, sizeof(float)* tet_number);
		cudaMemcpy(dev_tetVolumeOriginal, tetVolumeOriginal, sizeof(float)* tet_number, cudaMemcpyHostToDevice);
	}



	void computeShapeInfo(float ztop, float zbottom, int meshRes, float3 lensCenter, float lSemiMajorAxis, float lSemiWidth, float3 majorAxis, float3 lensDir)
	{
		float3 minorAxis = cross(lensDir, majorAxis);
		float3 rangeDiff = make_float3(2.1* lSemiMajorAxis, 2 * lSemiWidth, ztop - zbottom);
		float maxDiff = std::max(rangeDiff.x, std::max(rangeDiff.y, rangeDiff.z));

		//the following will make the mesh satisfy two conditions:
		//1, mimNumStepEachDim is satisfied
		//2, at least one dimension achieves the required resolution meshRes
		const int mimNumStepEachDim = 7;
		bool ismeshillegal = true;
		while (ismeshillegal){
			step = (maxDiff / meshRes);
			nStep[0] = floor(rangeDiff.x / step) + 1;
			nStep[1] = floor(rangeDiff.y / step) + 1;
			nStep[2] = floor(rangeDiff.z / step) + 1;
			if (nStep[0] >= mimNumStepEachDim && nStep[1] >= mimNumStepEachDim && nStep[2] >= mimNumStepEachDim){
				ismeshillegal = false;
				if (nStep[1] % 2 != 1){
					nStep[1]++;
				}
				rangeDiff.x = (nStep[0] - 1) * step;
				rangeDiff.y = (nStep[1] - 1) * step;
				rangeDiff.z = (nStep[2] - 1) * step;
			}
			else
			{
				meshRes++;
			}
		}

		std::cout << "final mesh size " << nStep[0] << " " << nStep[1] << " " << nStep[2] << ", with step length " << step << std::endl;
		
		number = nStep[0] * nStep[1] * nStep[2] + (nStep[0] - 2)*nStep[2];
		tet_number = (nStep[0] - 1) * (nStep[1] - 1) * (nStep[2] - 1) * 5;
		cutY = nStep[1] / 2;
		lensSpaceOriginInWorld = lensCenter - rangeDiff.x / 2.0 * majorAxis - rangeDiff.y / 2.0 * minorAxis + zbottom * lensDir;
	}

	void computeInitCoord(float3 majorAxis, float3 lensDir, glm::mat4 &meshTransMat)
	{

		float3 minorAxis = cross(lensDir, majorAxis);

		//find the matrix that transforms the coord in the the lens space to the world space
		//lens space is defined by majorAxis, minorAxis, and lensDir as x,y,z axis, and lensSpaceOriginInWorld as origin

		//rotate to fit the x axis to the majorAxis of the lens
		float3 rotateAxis = cross(majorAxis, make_float3(1, 0, 0));
		glm::mat4 r1 = glm::rotate(-(float)(acos(dot(make_float3(1, 0, 0), majorAxis))), glm::vec3(rotateAxis.x, rotateAxis.y, rotateAxis.z));

		//rotate to fit the z axis to the lenCenter-eye connecting line
		glm::vec4 lensDirByR1 = glm::inverse(r1)*glm::vec4(lensDir.x, lensDir.y, lensDir.z, 0);
		float3 rotateAxis2 = cross(make_float3(lensDirByR1.x, lensDirByR1.y, lensDirByR1.z), make_float3(0, 0, 1));
		glm::mat4 r2 = glm::rotate(-(float)(acos(dot(make_float3(0, 0, 1), make_float3(lensDirByR1.x, lensDirByR1.y, lensDirByR1.z)))), glm::vec3(rotateAxis2.x, rotateAxis2.y, rotateAxis2.z));

		//translate the origin to the lensSpaceOriginInWorld
		glm::vec4 lensSpaceOriginInWorldR1R2 = glm::inverse(r1*r2)*glm::vec4(lensSpaceOriginInWorld.x, lensSpaceOriginInWorld.y, lensSpaceOriginInWorld.z, 1);
		glm::mat4 t1 = glm::translate(glm::vec3(lensSpaceOriginInWorldR1R2));

		meshTransMat = r1*r2*t1;


		int idx;
		for (int i = 0; i < nStep[0]; i++){
			for (int j = 0; j <nStep[1]; j++){
				for (int k = 0; k < nStep[2]; k++){
					idx = i * nStep[1] * nStep[2] + j * nStep[2] + k;

					////code for drawing mesh with incision
					//glm::vec4 res;
					//if (j == cutY && i > 0 && i < nStep[0] - 1)
					//	res = meshTransMat*glm::vec4(i * step, j * step - step*0.1, k * step, 1.0f);
					//else
					//	res = meshTransMat*glm::vec4(i * step, j * step, k * step, 1.0f);

					glm::vec4 res = meshTransMat*glm::vec4(i * step, j * step, k * step, 1.0f);

					X[3 * idx + 0] = res.x;
					X[3 * idx + 1] = res.y;
					X[3 * idx + 2] = res.z;
				}
			}
		}
		for (int j = 0; j < 1; j++){
			for (int i = 0; i < nStep[0] - 2; i++){
				for (int k = 0; k < nStep[2]; k++){
					idx = nStep[0] * nStep[1] * nStep[2] + k* (nStep[0] - 2) + i;

					////code for drawing mesh with incision
					//glm::vec4 res = meshTransMat*glm::vec4((i + 1) * step, cutY * step + step*0.1, k * step, 1.0f);

					glm::vec4 res = meshTransMat*glm::vec4((i + 1) * step, cutY * step, k * step, 1.0f);

					X[3 * idx + 0] = res.x;
					X[3 * idx + 1] = res.y;
					X[3 * idx + 2] = res.z;
				}
			}
		}
	}



	void MoveMesh(float3 moveDir)
	{
		int threadsPerBlock = 64;
		int blocksPerGrid = (number + threadsPerBlock - 1) / threadsPerBlock;
		
		d_moveMeshNodes << <blocksPerGrid, threadsPerBlock >> >(dev_X, dev_X_Orig, number, moveDir);

		for (int i = 0; i < number; i++){	
			int idx = i * 3;
			X[idx + 0] = X[idx + 0] + moveDir.x;
			X[idx + 1] = X[idx + 1] + moveDir.y;
			X[idx + 2] = X[idx + 2] + moveDir.z;
		}
	}

	void UpdateMeshDevElasticity()
	{
		//does not work well. maybe more inner variables, which are decided by elasticity, need to be updated too
		cudaMemcpy(dev_EL, EL, sizeof(float)*tet_number, cudaMemcpyHostToDevice);
	}
};


#endif
