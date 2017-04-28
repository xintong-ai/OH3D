///////////////////////////////////////////////////////////////////////////////////////////
//  Class GridMesh
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef GRID_MESH_H
#define GRID_MESH_H
#include "physics/CUDA_PROJECTIVE_TET_MESH.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <algorithm>

inline int IdxConv(int idx, int nStep[3], int3 vc)
{
	return idx + vc.x * nStep[1] * nStep[2] + vc.y * nStep[2] + vc.z;
}

inline int IdxConv(int i, int j, int k, int nStep[3], int3 vc)
{
	return (i + vc.x) * nStep[1] * nStep[2] + (j + vc.y) * nStep[2] + k + vc.z;
}

template <class TYPE>
inline void Copy3(TYPE from[3], TYPE to[3])
{
	to[0] = from[0];
	to[1] = from[1];
	to[2] = from[2];
}

template <class TYPE>
class GridMesh: public CUDA_PROJECTIVE_TET_MESH<TYPE> 
{

  
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
	GridMesh(float dataMin[3], float dataMax[3], int n) : CUDA_PROJECTIVE_TET_MESH<TYPE>((n + 1) * (n + 1) * (n + 1) * 5)
	{
		float3 rangeDiff;
		float gridMin[3];
		float gridMax[3];
		float marginScale = 0.1;
		for (int i = 0; i < 3; i++){
			float marginSize = (dataMax[i] - dataMin[i]) * marginScale;
			gridMin[i] = dataMin[i] - marginSize;
			gridMax[i] = dataMax[i] + marginSize;
		}
		rangeDiff = make_float3(
			gridMax[0] - gridMin[0],
			gridMax[1] - gridMin[1],
			gridMax[2] - gridMin[2]);
		float maxDiff = std::max(rangeDiff.x, std::max(rangeDiff.y, rangeDiff.z));
		step = (maxDiff / n) * 1.01;
		//Read_Original_File("sorted_armadillo");
		//Read_Original_File("armadillo_10k.1");
		//float dataMin[3] = { -0.50, -0.50, -0.50 };
		//float dataMax[3] = { 0.50, 0.50, 0.50 };
		BuildMesh(gridMin, gridMax, step);
		//Scale(0.008);
		//Centralize(); 
		printf("N: %d, %d\n", number, tet_number);

		//Set fixed nodes
		//Rotate_X(-0.2);
		//for(int v=0; v<number; v++)
		//	//if(X[v*3+1]>-0.04 && X[v*3+1]<0)		
		//	//	if(fabsf(X[v*3+1]+0.01)<1*(X[v*3+2]-0.1))
		//	if(fabsf(X[v*3+1]+0.01)<2*(X[v*3+2]-0.2))
		//		fixed[v]=10000000;
		//Rotate_X(1.2);

		//for(int v=0; v<number; v++)
		//	if (X[v * 3 + 0] < (gridMin.x + 0.0001) || X[v * 3 + 0] > (gridMax.x - 0.0001)
		//		|| X[v * 3 + 1] < (gridMin.y + 0.0001) || X[v * 3 + 1] > (gridMax.x - 0.0001)
		//		|| X[v * 3 + 2] < (gridMin.z + 0.0001) || X[v * 3 + 2] > (gridMax.x - 0.0001))
		//		fixed[v]=10000000;

		//elasticity = 1800;// 18000000; //5000000
		control_mag	= 500;		//500
		damping = 0.9;
	}

};


#endif
