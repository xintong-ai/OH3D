#include <ModelGrid.h>
#include <GridMesh.h>
#include <LineSplitGridMesh.h>

#include <defines.h>

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
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		gridMesh = new GridMesh<float>(dmin, dmax, n);
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		lsgridMesh = new LineSplitGridMesh<float>(dmin, dmax, n); //can be removed later. may only init when needed. currently just to make gridRenderable not crash
}

ModelGrid::ModelGrid(float dmin[3], float dmax[3], int n, bool useLineSplitGridMesh)
{
	_n = n;
	_dmin[0] = dmin[0];
	_dmin[1] = dmin[1];
	_dmin[2] = dmin[2];
	_dmax[0] = dmax[0];
	_dmax[1] = dmax[1];
	_dmax[2] = dmax[2];

	if (useLineSplitGridMesh)
		gridType = LINESPLIT_UNIFORM_GRID;
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		gridMesh = new GridMesh<float>(dmin, dmax, n);
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		lsgridMesh = new LineSplitGridMesh<float>(dmin, dmax, n);

}

float3 regularMeshVertCoord(int iv, int3 nStep, int cutY, float3 gridMin, float step){
	int i = iv / (nStep.y * nStep.z);
	int j = (iv - i* nStep.y * nStep.z) / nStep.z;
	int k = iv - i * nStep.y * nStep.z - j * nStep.z;
	if (j <= cutY){
		return make_float3(gridMin.x + i * step, gridMin.y + j * step, gridMin.z + k * step);
	}
	else{
		return make_float3(gridMin.x + i * step, gridMin.y + (j - 1 + 0.01) * step, gridMin.z + k * step);
	}

}


void ModelGrid::UpdatePointTetId(float4* v, int n)
{
	glm::mat4 invMeshTransMat = glm::inverse(meshTransMat);
	vIdx.resize(n);
	vBaryCoord.resize(n);

	float3 gridMin = GetGridMin();
	float3 gridMax = GetGridMax();
	int3 nStep = GetNumSteps();
	float step = GetStep();
	int* tet = GetTet();
	//float* X = GetX();

	for (int i = 0; i < n; i++){
		glm::vec4 g_vcOri = glm::vec4(v[i].x, v[i].y, v[i].z, 1.0);
		glm::vec4 g_vcTransformed = invMeshTransMat*g_vcOri;

		float3 vc = make_float3(g_vcTransformed.x, g_vcTransformed.y, g_vcTransformed.z);
		float3 tmp = (vc - gridMin) / step;
		int3 idx3 = make_int3(tmp.x, tmp.y, tmp.z);
		if (idx3.y >= lsgridMesh->cutY) //has a slight error
			idx3.y++;

		if (idx3.x < 0 || idx3.y < 0 || idx3.z < 0 || idx3.x >= nStep.x - 1 || idx3.y >= nStep.y - 1 || idx3.z >= nStep.z - 1){
			vIdx[i] = -1;
		}
		else{
			int idx = idx3.x * (nStep.y - 1) * (nStep.z - 1)
				+ idx3.y * (nStep.z - 1) + idx3.z;

			int j = 0;
			for (j = 0; j < 5; j++){
				float3 vv[4];
				int tetId = idx * 5 + j;
				for (int k = 0; k < 4; k++){
					int iv = tet[tetId * 4 + k];



					vv[k] = regularMeshVertCoord(iv, nStep, lsgridMesh->cutY, gridMin, step);
					//vv[k] = make_float3(X[3 * iv + 0], X[3 * iv + 1], X[3 * iv + 2]);
				}
				float4 bary = GetBarycentricCoordinate(vv[0], vv[1], vv[2], vv[3], vc);
				if (within(bary.x) && within(bary.y) && within(bary.z) && within(bary.w)) {
					vIdx[i] = idx * 5 + j;
					vBaryCoord[i] = bary;
					break;
				}
			}
			if (j == 6){ //need to be handle later
				vIdx[i] = -1;
			}
		}
	}
}


void ModelGrid::UpdatePointCoords(float4* v, int n, float4* vOri, bool needUpdateTetId)
{


	if (needUpdateTetId) { //whether need to update the tet ids of the points
		UpdatePointTetId(v, n);
	}

	//for (int i = 0; i < n; i++){

	//	v[i] = vOri[i];
	//}
	//return;

	int tet_number = GetTNumber();

	int* tet = GetTet();
	float* X = GetX();
	for (int i = 0; i < n; i++){
		int vi = vIdx[i];
		if (vi >= 0 && vi < tet_number){
			float4 vb = vBaryCoord[i];
			float3 vv[4];
			for (int k = 0; k < 4; k++){
				int iv = tet[vi * 4 + k];
				vv[k] = make_float3(X[3 * iv + 0], X[3 * iv + 1], X[3 * iv + 2]);
			}
			v[i] = make_float4(vb.x * vv[0] + vb.y * vv[1] + vb.z * vv[2] + vb.w * vv[3], 1);
		}
		else if (vOri != 0){
			v[i] = vOri[i];
		}
		//others should not happen
	}
}




void ModelGrid::InitGridDensity(float4* v, int n)
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
	{

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
}


void ModelGrid::ReinitiateMesh(float3 lensCenter, float lSemiMajorAxis, float lSemiMinorAxis, float3 direction, float focusRatio, float3 negZAxisClipInGlobal)
{
	if (!bMeshNeedReinitiation)
		return;

	if (gridType != GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return;

	//std::cout << "lensCenter " << lensCenter.x << " " << lensCenter.y << " " << lensCenter.z
	//	<< std::endl;

	//std::cout << "lSemiMajorAxis " << lSemiMajorAxis << std::endl;
	//std::cout << "lSemiMinorAxis " << lSemiMinorAxis << std::endl;
	//std::cout << "direction " << direction.x << " " << direction.y << " " << direction.z
	//	<< std::endl;

	if (lsgridMesh != 0)
		delete lsgridMesh;
	lsgridMesh = new LineSplitGridMesh<float>(_dmin, _dmax, _n, lensCenter, lSemiMajorAxis, lSemiMinorAxis, direction, focusRatio, negZAxisClipInGlobal, meshTransMat);
	SetElasticitySimple();
	lsgridMesh->Initialize(time_step);

	bMeshNeedReinitiation = false;
}



void ModelGrid::SetElasticitySimple()
{
	std::vector<float> density;
	density.resize(lsgridMesh->tet_number);
	for (int i = 0; i < density.size(); i++) {
		density[i] = 10.1; //the higher density, the more deformation is reached
	}
	std::copy(&density[0], &density[0] + lsgridMesh->tet_number, lsgridMesh->EL);
}

void ModelGrid::Initialize(float time_step)
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		gridMesh->Initialize(time_step);
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		lsgridMesh->Initialize(time_step);
	else
		return;
}

void ModelGrid::Update(float lensCenter[3], float lenDir[3], float focusRatio, float radius)
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		gridMesh->Update(time_step, 64, lensCenter, lenDir, focusRatio, radius);
	return;
}

void ModelGrid::Update(float lensCenter[3], float lenDir[3], float lSemiMajorAxis, float lSemiMinorAxis, float focusRatio, float radius, float3 majorAxisGlobal)
{
	if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		lsgridMesh->Update(time_step, 1, lensCenter, lenDir, lsgridMesh->meshCenter, lsgridMesh->cutY, lsgridMesh->nStep, lSemiMajorAxis, lSemiMinorAxis, focusRatio, radius, majorAxisGlobal);
	return;
}

/////////////////////////////////////// attributes getters /////////////////////

int ModelGrid::GetTNumber()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->t_number;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->t_number;
	else
		return -1;
}

int* ModelGrid::GetT()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->T;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->T;
	else
		return NULL;
}

float* ModelGrid::GetX()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->X;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->X;
	else
		return NULL;
}

int ModelGrid::GetNumber()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->number;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->number;
	else
		return 0;
}

unsigned int* ModelGrid::GetL()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->L;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->L;
	else
		return NULL;
}

float* ModelGrid::GetE()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->EL;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->EL;
	else
		return NULL;
}

int ModelGrid::GetLNumber()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->l_number;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->l_number;
	else
		return 0;
}

float3 ModelGrid::GetGridMin()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->gridMin;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->gridMin;
	else
		return make_float3(0,0,0);
}

float3 ModelGrid::GetGridMax()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->gridMax;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->gridMax;
	else
		return make_float3(0, 0, 0);
}

int3 ModelGrid::GetNumSteps()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return make_int3(gridMesh->nStep[0], gridMesh->nStep[1], gridMesh->nStep[2]);
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return make_int3(lsgridMesh->nStep[0], lsgridMesh->nStep[1], lsgridMesh->nStep[2]);
	else
		return make_int3(0, 0, 0);
}

float ModelGrid::GetStep()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->step;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->step;
	else
		return 0;
}

void ModelGrid::SetElasticity(float* v)
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		std::copy(v, v + gridMesh->tet_number, gridMesh->EL);
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		std::copy(v, v + lsgridMesh->tet_number, lsgridMesh->EL);

}

int* ModelGrid::GetTet()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->Tet;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->Tet;
	else
		return NULL;
}

int ModelGrid::GetTetNumber()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->tet_number;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->tet_number;
	else
		return NULL;
}
