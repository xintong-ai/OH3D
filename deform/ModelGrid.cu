#include <ModelGrid.h>
#include <GridMesh.h>
#include <ModelLineSplitGrid.h>

//the algorithm of converting from Cartesian to Barycentric coordinates is from
//http://dennis2society.de/painless-tetrahedral-barycentric-mapping
/**
* Calculate the determinant for a 4x4 matrix based on this example:
* http://www.euclideanspace.com/maths/algebra/matrix/functions/determinant/fourD/index.htm
* This function takes four float4 as row vectors and calculates the resulting matrix' determinant
* using the Laplace expansion.
*
*/


inline bool within(float v)
{
	return v >= 0 && v <= 1;
}


const float Determinant4x4(const float4& v0,
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

/**
* Calculate the actual barycentric coordinate from a point p0_ and the four
* vertices v0_ .. v3_ from a tetrahedron.
*/
const float4 GetBarycentricCoordinate(const float3& v0_,
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
	const float det0 = Determinant4x4(v0, v1, v2, v3);
	const float det1 = Determinant4x4(p0, v1, v2, v3);
	const float det2 = Determinant4x4(v0, p0, v2, v3);
	const float det3 = Determinant4x4(v0, v1, p0, v3);
	const float det4 = Determinant4x4(v0, v1, v2, p0);
	barycentricCoord.x = (det1 / det0);
	barycentricCoord.y = (det2 / det0);
	barycentricCoord.z = (det3 / det0);
	barycentricCoord.w = (det4 / det0);
	return barycentricCoord;
}

ModelGrid::ModelGrid(float dmin[3], float dmax[3], int n)
{
	gridMesh = new GridMesh<float>(dmin, dmax, n);
}

void ModelGrid::UpdatePointCoords(float4* v, int n)
{
	int* tet = GetTet();
	float* X = GetX();
	for (int i = 0; i < n; i++){
		int vi = vIdx[i];
		float4 vb = vBaryCoord[i];
		float3 vv[4];
		for (int k = 0; k < 4; k++){
			int iv = tet[vi * 4 + k];
			vv[k] = make_float3(X[3 * iv + 0], X[3 * iv + 1], X[3 * iv + 2]);
		}
		v[i] = make_float4(vb.x * vv[0] + vb.y * vv[1] + vb.z * vv[2] + vb.w * vv[3], 1);
	}
}


int ModelGrid::GetTNumber()
{
	return gridMesh->t_number;
}

void ModelGrid::InitGridDensity(float4* v, int n)
{
	//;
	float3 gridMin = GetGridMin();
	float3 gridMax = GetGridMax();
	int3 nStep = GetNumSteps();
	float step = GetStep();
	int* tet = GetTet();
	float* X = GetX();
	std::vector<int> cnts;
	cnts.resize((nStep.x - 1) *(nStep.y - 1) *(nStep.z - 1) * 5, 0);
	vBaryCoord.resize(n);
	vIdx.resize(n);
	for (int i = 0; i < n; i++){
		float3 vc = make_float3(v[i].x, v[i].y, v[i].z);
		float3 tmp = (vc - gridMin) / step;
		int3 idx3 = make_int3(tmp.x, tmp.y, tmp.z);
		int idx = idx3.x * (nStep.y - 1) * (nStep.z - 1)
			+ idx3.y * (nStep.z - 1) + idx3.z;
		for (int j = 0; j < 5; j++){
			float3 vv[4];
			int tetId = idx * 5 + j;
			for (int k = 0; k < 4; k++){
				int iv = tet[tetId * 4 + k];
				vv[k] = make_float3(X[3 * iv + 0], X[3 * iv + 1], X[3 * iv + 2]);
			}
			float4 bary = GetBarycentricCoordinate(vv[0], vv[1], vv[2], vv[3], vc);
			if (within(bary.x) && within(bary.y) && within(bary.z) && within(bary.w)) {
				vIdx[i] = idx * 5 + j;
				vBaryCoord[i] = bary;
				cnts[tetId]++;
				break;
			}
		}
		//float3 local = make_float3(tmp.x - idx3.x, tmp.y - idx3.y, tmp.z - idx3.z);// vc - (gridMin + make_float3(idx3.x * step, idx3.y * step, idx3.z * step));

		//localCoord.push_back(make_float4(idx, local.x, local.y, local.z));
	}
	std::vector<float> density;
	density.resize(cnts.size());
	//const float base = 400.0f / cnts.size();
	for (int i = 0; i < cnts.size(); i++) {
		//for (int j = 0; j < 5; j++) {
		density[i] = 500 + 1000 * pow((float)cnts[i], 2);
		//}
	}
	SetElasticity(&density[0]);
	Initialize(time_step);
}

int* ModelGrid::GetT()
{
	return gridMesh->T;
}

float* ModelGrid::GetX()
{
	return gridMesh->X;
}

int ModelGrid::GetNumber()
{
	return gridMesh->number;
}

unsigned int* ModelGrid::GetL()
{
	return gridMesh->L;
}

float* ModelGrid::GetE()
{
	return gridMesh->EL;
}


int ModelGrid::GetLNumber()
{
	return gridMesh->l_number;
}

void ModelGrid::Initialize(float time_step)
{
	return gridMesh->Initialize(time_step);
}

void ModelGrid::Update(float lensCenter[3], float lenDir[3], float focusRatio, float radius)
{
	gridMesh->Update(time_step, 64, lensCenter, lenDir, focusRatio, radius);
}

float3 ModelGrid::GetGridMin()
{
	return gridMesh->gridMin;
}

float3 ModelGrid::GetGridMax()
{
	return gridMesh->gridMax;
}

int3 ModelGrid::GetNumSteps()
{
	return make_int3(gridMesh->nStep[0], gridMesh->nStep[1], gridMesh->nStep[2]);
}

float ModelGrid::GetStep()
{
	return gridMesh->step;
}

void ModelGrid::SetElasticity(float* v)
{
	std::copy(v, v + gridMesh->tet_number, gridMesh->EL);
}

int* ModelGrid::GetTet()
{
	return gridMesh->Tet;
}
