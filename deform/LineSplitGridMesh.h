///////////////////////////////////////////////////////////////////////////////////////////
//  Class LineSplitGridMesh
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef LINESPLIT_GRID_MESH_H
#define LINESPLIT_GRID_MESH_H
#include "physics/CUDA_PROJECTIVE_TET_MESH.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <algorithm>

#include "D:\Library\OpenGL\glm\glm\glm.hpp"
#include <D:/Library/OpenGL/glm/glm/gtx/transform.hpp>

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
	using TET_MESH<TYPE>::tet_number;
	using TET_MESH<TYPE>::number;
	using TET_MESH<TYPE>::Tet;
	using TET_MESH<TYPE>::inv_Dm;
	using TET_MESH<TYPE>::Vol;
	using TET_MESH<TYPE>::X;
	using TET_MESH<TYPE>::Dm;
	using CUDA_PROJECTIVE_TET_MESH<TYPE>::control_mag;
	using TET_MESH<TYPE>::max_number;
	using TET_MESH<TYPE>::M;
	using TET_MESH<TYPE>::t_number;
	using TET_MESH<TYPE>::T;
	using TET_MESH<TYPE>::VN;
	using TET_MESH<TYPE>::TN;
	using TET_MESH<TYPE>::l_number;
	using TET_MESH<TYPE>::L;
	using CUDA_PROJECTIVE_TET_MESH<TYPE>::damping;


	float3 gridMin, gridMax;
	int nStep[3];
	float step;

	float3 meshCenter;
	int cutY;
	float3 oriMeshCenter;

	float3 lensSpaceOriginInWorld;

	void Build_Boundary_Triangles2()
	{
		t_number = tet_number * 4;
		for (int i = 0; i < tet_number; i++) {
			T[i * 4 * 3 + 0 * 3 + 0] = Tet[i * 4 + 0];
			T[i * 4 * 3 + 0 * 3 + 1] = Tet[i * 4 + 1];
			T[i * 4 * 3 + 0 * 3 + 2] = Tet[i * 4 + 2];

			T[i * 4 * 3 + 1 * 3 + 0] = Tet[i * 4 + 0];
			T[i * 4 * 3 + 1 * 3 + 1] = Tet[i * 4 + 1];
			T[i * 4 * 3 + 1 * 3 + 2] = Tet[i * 4 + 3];

			T[i * 4 * 3 + 2 * 3 + 0] = Tet[i * 4 + 0];
			T[i * 4 * 3 + 2 * 3 + 1] = Tet[i * 4 + 2];
			T[i * 4 * 3 + 2 * 3 + 2] = Tet[i * 4 + 3];

			T[i * 4 * 3 + 3 * 3 + 0] = Tet[i * 4 + 1];
			T[i * 4 * 3 + 3 * 3 + 1] = Tet[i * 4 + 2];
			T[i * 4 * 3 + 3 * 3 + 2] = Tet[i * 4 + 3];
		}
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

	void computeShapeInfo(float dataMin[3], float dataMax[3], int n)
	{
		float3 rangeDiff;
		float gridMinInit[3];
		float gridMaxInit[3];
		//float marginScale = 0.1;
		float marginScale = (length(make_float3(dataMax[0], dataMax[1], dataMax[2]) - make_float3(dataMin[0], dataMin[1], dataMin[2])) / (dataMax[2] - dataMin[2]) - 1) / 2; //make sure the z-dim of the mesh can cover the volume all the time
		for (int i = 0; i < 3; i++){
			float marginSize = (dataMax[i] - dataMin[i]) * marginScale;
			gridMinInit[i] = dataMin[i] - marginSize;
			gridMaxInit[i] = dataMax[i] + marginSize;
		}
		rangeDiff = make_float3(
			gridMaxInit[0] - gridMinInit[0],
			gridMaxInit[1] - gridMinInit[1],
			gridMaxInit[2] - gridMinInit[2]);
		float maxDiff = std::max(rangeDiff.x, std::max(rangeDiff.y, rangeDiff.z));
		//step = (maxDiff / n) * 1.01;
		step = (maxDiff / n);

		for (int i = 0; i < 3; i++){
			nStep[i] = ceil((gridMaxInit[i] - gridMinInit[i]) / step) + 1;
		}
		number = nStep[0] * nStep[1] * nStep[2] + (nStep[0] - 2)*nStep[2];
		gridMin = make_float3(gridMinInit[0], gridMinInit[1], gridMinInit[2]);
		gridMax = make_float3(gridMinInit[0] + (nStep[0] - 1) * step, gridMinInit[1] + (nStep[1] - 1) * step, gridMinInit[2] + (nStep[2] - 1) * step);

		cutY = nStep[1] / 2;
		oriMeshCenter = make_float3((gridMin.x + gridMax.x) / 2, gridMin.y + cutY * step, (gridMin.z + gridMax.z) / 2);	
	}

	void BuildTet()
	{
		tet_number = (nStep[0] - 1) * (nStep[1] - 1) * (nStep[2] - 1) * 5;
		
		tetVolume = new float[tet_number];
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
						
						tetVolume[idx * 5 + 0] = tetVolumeCandidate1;
						tetVolume[idx * 5 + 1] = tetVolumeCandidate1;
						tetVolume[idx * 5 + 2] = tetVolumeCandidate1;
						tetVolume[idx * 5 + 3] = tetVolumeCandidate1;
						tetVolume[idx * 5 + 4] = tetVolumeCandidate2;

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
						
						tetVolume[idx * 5 + 0] = tetVolumeCandidate1;
						tetVolume[idx * 5 + 1] = tetVolumeCandidate1;
						tetVolume[idx * 5 + 2] = tetVolumeCandidate1;
						tetVolume[idx * 5 + 3] = tetVolumeCandidate1;
						tetVolume[idx * 5 + 4] = tetVolumeCandidate2; 
						
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
	}

	//template <class TYPE>
	void BuildMesh(TYPE dmin[3], TYPE dmax[3], TYPE step)
	{
		for (int i = 0; i < 3; i++){
			nStep[i] = ceil((dmax[i] - dmin[i]) / step) + 1;
		}
		number = nStep[0] * nStep[1] * nStep[2];
		int idx = 0;
		//int end[3] = {0, 0, 0};// whether it is at the two ends of the axis
		for (int i = 0; i < nStep[0]; i++){
			//end[0] = ((i == 0) || (i == (nStep[0] - 1))) ? 1 : 0;
			for (int j = 0; j < nStep[1]; j++){
				//end[1] = ((j == 0) || (j == (nStep[1] - 1))) ? 1 : 0;
				for (int k = 0; k < nStep[2]; k++){
					//end[2] = ((k == 0) || (k == (nStep[2] - 1))) ? 1 : 0;
					idx = i * nStep[1] * nStep[2] + j * nStep[2] + k;
					X[3 * idx + 0] = dmin[0] + i * step;
					X[3 * idx + 1] = dmin[1] + j * step;
					X[3 * idx + 2] = dmin[2] + k * step;
					//if ((end[0] + end[1] + end[2]) > 0)
					//fixed[idx] = 10000000;
				}
			}
		}
		gridMin = make_float3(dmin[0], dmin[1], dmin[2]);
		gridMax = make_float3(dmin[0] + (nStep[0] - 1) * step, dmin[1] + (nStep[1] - 1) * step, dmin[2] + (nStep[2] - 1) * step);

		int3 vc[8];
		vc[0] = make_int3(0, 0, 0);
		vc[1] = make_int3(1, 0, 0);
		vc[2] = make_int3(0, 1, 0);
		vc[3] = make_int3(1, 1, 0);
		vc[4] = make_int3(0, 0, 1);
		vc[5] = make_int3(1, 0, 1);
		vc[6] = make_int3(0, 1, 1);
		vc[7] = make_int3(1, 1, 1);

		int idx2 = 0;
		tet_number = (nStep[0] - 1) * (nStep[1] - 1) * (nStep[2] - 1) * 5;
		for (int i = 0; i < (nStep[0] - 1); i++){
			for (int j = 0; j < (nStep[1] - 1); j++){
				for (int k = 0; k < (nStep[2] - 1); k++){
					idx = i * (nStep[1] - 1) * (nStep[2] - 1) + j * (nStep[2] - 1) + k;
					idx2 = i * nStep[1] * nStep[2] + j * nStep[2] + k;
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
		//Build_Boundary_Triangles2();
		Build_Boundary_Lines();
		//PrintMesh();
	}

	//template <class TYPE>
	LineSplitGridMesh(float dataMin[3], float dataMax[3], int n) : CUDA_PROJECTIVE_TET_MESH<TYPE>((n + 1) * (n + 1) * (n + 1)*5)
	{

		computeShapeInfo(dataMin, dataMax, n);
		BuildTet();
		Build_Boundary_Lines();


		printf("N: %d, %d\n", number, tet_number);

		control_mag = 500;		//500
		damping = 0.9;
		return;
	}

	void computeInitCoord(float3 lensCenter, float lSemiMajorAxis, float lSemiMinorAxis, float3 majorAxis, float focusRatio, float3 lensDir, glm::mat4 &meshTransMat)
	{
		//rotate to fit the x axis to the majorAxis of the lens
		float3 rotateAxis = cross(make_float3(1, 0, 0), majorAxis);
		glm::mat4 r1 = glm::rotate((float)(acos(dot(make_float3(1, 0, 0), majorAxis))), glm::vec3(rotateAxis.x, rotateAxis.y, rotateAxis.z));

		//rotate to fit the z axis to the lenCenter-eye connecting line
		glm::vec4 lensDirByR1 = glm::inverse(r1)*glm::vec4(lensDir.x, lensDir.y, lensDir.z, 0);
		float3 rotateAxis2 = cross(make_float3(0, 0, 1), make_float3(lensDirByR1.x, lensDirByR1.y, lensDirByR1.z));
		glm::mat4 r2 = glm::rotate((float)(acos(dot(make_float3(0, 0, 1), make_float3(lensDirByR1.x, lensDirByR1.y, lensDirByR1.z)))), glm::vec3(rotateAxis2.x, rotateAxis2.y, rotateAxis2.z));

		//translate the mesh center to the point which is on the lenCenter-eye connecting line and the closet to the data center, estimated by gridMin and gridMax
		float3 dataCenter = (gridMin + gridMax);
		float3 translateTarget = lensCenter + dot(dataCenter - lensCenter, lensDir)*lensDir;
		//glm::vec4 lensCenterByR1R2 = glm::inverse(r1*r2)*glm::vec4(lensCenter.x, lensCenter.y, lensCenter.z, 1);
		glm::vec4 lensCenterByR1R2 = glm::inverse(r1*r2)*glm::vec4(translateTarget.x, translateTarget.y, translateTarget.z, 1);
		glm::mat4 t1 = glm::translate(glm::vec3(lensCenterByR1R2.x - oriMeshCenter.x, lensCenterByR1R2.y - oriMeshCenter.y, lensCenterByR1R2.z - oriMeshCenter.z));

		meshTransMat = r1*r2*t1;

		int idx;
		for (int i = 0; i < nStep[0]; i++){
			for (int j = 0; j <nStep[1]; j++){
				for (int k = 0; k < nStep[2]; k++){
					idx = i * nStep[1] * nStep[2] + j * nStep[2] + k;

					glm::vec4 res = meshTransMat*glm::vec4(gridMin.x + i * step, gridMin.y + j * step, gridMin.z + k * step, 1.0f);

					X[3 * idx + 0] = res.x;
					X[3 * idx + 1] = res.y;
					X[3 * idx + 2] = res.z;
				}
			}
		}
		for (int j = 0; j < 1; j++){
			for (int i = 0; i < nStep[0]-2; i++){
				for (int k = 0; k < nStep[2]; k++){
					idx = nStep[0] * nStep[1] * nStep[2] + k* (nStep[0]-2) + i;

					glm::vec4 res = meshTransMat*glm::vec4(gridMin.x + (i+1) * step, gridMin.y + cutY * step , gridMin.z + k * step, 1.0f);

					X[3 * idx + 0] = res.x;
					X[3 * idx + 1] = res.y;
					X[3 * idx + 2] = res.z;
				}
			}
		}
	
	}
	

	void computeShapeInfo2(float dataMin[3], float dataMax[3], int n, float3 lensCenter, float lSemiMajorAxis, float lSemiMinorAxis, float3 majorAxis, float focusRatio, float3 lensDir)
	{
		float volumeCornerx[2], volumeCornery[2], volumeCornerz[2];
		volumeCornerx[0] = dataMin[0];
		volumeCornery[0] = dataMin[1];
		volumeCornerz[0] = dataMin[2];
		volumeCornerx[1] = dataMax[0];
		volumeCornery[1] = dataMax[1];
		volumeCornerz[1] = dataMax[2];

		float rx1, rx2, ry1, ry2;
		float rz1, rz2;//at the direction lensDir shooting from lensCenter, the range to cover the whole region
		rx1 = FLT_MAX, rx2 = -FLT_MAX;
		ry1 = FLT_MAX, ry2 = -FLT_MAX;
		rz1 = FLT_MAX, rz2 = -FLT_MAX;

		float3 minorAxis = cross(lensDir, majorAxis);

		//currently computing r1 and r2 aggressively. cam be more improved later
		for (int k = 0; k < 2; k++){
			for (int j = 0; j < 2; j++){
				for (int i = 0; i < 2; i++){
					float3 dir = make_float3(volumeCornerx[i], volumeCornery[j], volumeCornerz[k]) - lensCenter;
					//float curx = dot(dir, majorAxis);
					//if (curx < rx1)
					//	rx1 = curx;
					//if (curx > rx2)
					//	rx2 = curx;

					//float cury = dot(dir, minorAxis);
					//if (cury < ry1)
					//	ry1 = cury;
					//if (cury > ry2)
					//	ry2 = cury;

					float curz = dot(dir, lensDir);
					if (curz < rz1)
						rz1 = curz;
					if (curz > rz2)
						rz2 = curz;
				}
			}
		}


		float3 rangeDiff = make_float3(3 * lSemiMajorAxis, 3 * lSemiMinorAxis / focusRatio, rz2 - rz1);
		float maxDiff = std::max(rangeDiff.x, std::max(rangeDiff.y, rangeDiff.z));

		bool ismeshillegal = true;
		while (ismeshillegal){
			step = (maxDiff / n);
			nStep[0] = floor(rangeDiff.x / step) + 1;
			nStep[1] = floor(rangeDiff.y / step) + 1;
			nStep[2] = floor(rangeDiff.z / step) + 1;
			if (nStep[0] >= 4 && nStep[1] >= 4 && nStep[2] >= 4){
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
				n++;
			}
		}
		std::cout << "final mesh size " << nStep[0] << " " << nStep[1] << " " << nStep[2] << std::endl;
		
		number = nStep[0] * nStep[1] * nStep[2] + (nStep[0] - 2)*nStep[2];
		
		cutY = nStep[1] / 2;

		lensSpaceOriginInWorld = lensCenter - rangeDiff.x / 2.0 * majorAxis - rangeDiff.y / 2.0 * minorAxis - rangeDiff.z / 2.0 * lensDir;

		//gridMin = make_float3(gridMinInit[0], gridMinInit[1], gridMinInit[2]);
		//gridMax = make_float3(gridMinInit[0] + (nStep[0] - 1) * step, gridMinInit[1] + (nStep[1] - 1) * step, gridMinInit[2] + (nStep[2] - 1) * step);

		//oriMeshCenter = make_float3((gridMin.x + gridMax.x) / 2, gridMin.y + cutY * step, (gridMin.z + gridMax.z) / 2);
	}

	void computeInitCoord2(float3 lensCenter, float lSemiMajorAxis, float lSemiMinorAxis, float3 majorAxis, float focusRatio, float3 lensDir, glm::mat4 &meshTransMat)
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

					glm::vec4 res = meshTransMat*glm::vec4((i + 1) * step, cutY * step, k * step, 1.0f);

					X[3 * idx + 0] = res.x;
					X[3 * idx + 1] = res.y;
					X[3 * idx + 2] = res.z;
				}
			}
		}


	}


	LineSplitGridMesh(float dataMin[3], float dataMax[3], int n, float3 lensCenter, float lSemiMajorAxis, float lSemiMinorAxis, float3 majorAxis, float focusRatio, float3 lensDir, glm::mat4 &meshTransMat) : CUDA_PROJECTIVE_TET_MESH<TYPE>(((n + 1) * (n + 1) * (n + 1) +(n+1)*(n-1))* 5)
	{
		//computeShapeInfo2 and computeShapeInfo2 define a mesh that covers the lens region and nearby region
		//computeShapeInfo and computeShapeInfo are old settings that define a mesh by the data domain
		computeShapeInfo2(dataMin, dataMax, n, lensCenter, lSemiMajorAxis, lSemiMinorAxis, majorAxis, focusRatio, lensDir);
		computeInitCoord2(lensCenter, lSemiMajorAxis, lSemiMinorAxis, majorAxis, focusRatio, lensDir, meshTransMat);

		BuildTet();
		Build_Boundary_Lines();


		printf("N: %d, %d\n", number, tet_number);

		control_mag = 500;		//500
		damping = 0.9;
		
		Allocate_GPU_Memory_InAdvance();
		is_Allocate_GPU_Memory_InAdvance_executed = true;
		
		return;

	}
};


#endif
