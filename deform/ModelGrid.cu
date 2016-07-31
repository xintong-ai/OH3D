#include <ModelGrid.h>
#include <GridMesh.h>
#include <LineSplitGridMesh.h>

#include <defines.h>
#include <Volume.h>


//the algorithm of converting from Cartesian to Barycentric coordinates is from
//http://dennis2society.de/painless-tetrahedral-barycentric-mapping
/**
* Calculate the determinant for a 4x4 matrix based on this example:
* http://www.euclideanspace.com/maths/algebra/matrix/functions/determinant/fourD/index.htm
* This function takes four float4 as row vectors and calculates the resulting matrix' determinant
* using the Laplace expansion.
*
*/

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <TransformFunc.h>

texture<float, 3, cudaReadModeElementType>  volumeTex;

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

	volumeTex.normalized = false;
	volumeTex.filterMode = cudaFilterModeLinear;
	volumeTex.addressMode[0] = cudaAddressModeBorder;
	volumeTex.addressMode[1] = cudaAddressModeBorder;
	volumeTex.addressMode[2] = cudaAddressModeBorder;
}

float3 regularMeshVertCoord(int iv, int3 nStep, float3 gridMin, float step){
	//int i = iv / (nStep.y * nStep.z);
	//int j = (iv - i* nStep.y * nStep.z) / nStep.z;
	//int k = iv - i * nStep.y * nStep.z - j * nStep.z;
	int x, y, z;

	if (iv < nStep.x * nStep.y * nStep.z){
		x = iv / (nStep.y * nStep.z);
		y = (iv - x* nStep.y * nStep.z) / nStep.z;
		z = iv - x * nStep.y * nStep.z - y * nStep.z;
	}
	else{
		int extra = iv - nStep.x * nStep.y * nStep.z;
		y = nStep.y / 2; // always == cutY
		z = extra / (nStep.x - 2);
		x = extra - z*(nStep.x) + 1;
	}

	return make_float3(gridMin.x + x * step, gridMin.y + y * step, gridMin.z + z * step);
}


//used for the mesh built based on lens region
void ModelGrid::UpdatePointTetId(float4* v, int n)
{
	glm::mat4 invMeshTransMat = glm::inverse(meshTransMat);
	vIdx.resize(n);
	vBaryCoord.resize(n);

	int3 nStep = GetNumSteps();
	float step = GetStep();
	int* tet = GetTet();
	float* X = GetX();

	for (int i = 0; i < n; i++){
		glm::vec4 g_vcOri = glm::vec4(v[i].x, v[i].y, v[i].z, 1.0);
		glm::vec4 g_vcTransformed = invMeshTransMat*g_vcOri;

		float3 vc = make_float3(g_vcTransformed.x, g_vcTransformed.y, g_vcTransformed.z);
		float3 tmp = vc / step;
		int3 idx3 = make_int3(floor(tmp.x), floor(tmp.y), floor(tmp.z));


		if (idx3.x < 0 || idx3.y < 0 || idx3.z < 0 || idx3.x >= nStep.x - 1 || idx3.y >= nStep.y - 1 || idx3.z >= nStep.z - 1){
			vIdx[i] = -1;
		}
		else{
			int cubeIdx = idx3.x * (nStep.y - 1) * (nStep.z - 1)
				+ idx3.y * (nStep.z - 1) + idx3.z;

			int j = 0;
			for (j = 0; j < 5; j++){
				float3 vv[4];
				int tetId = cubeIdx * 5 + j;
				for (int k = 0; k < 4; k++){
					int iv = tet[tetId * 4 + k];

					//vv[k] = regularMeshVertCoord(iv, nStep, gridMin, step);
					//vv[k] = make_float3(X[3 * iv + 0], X[3 * iv + 1], X[3 * iv + 2]);

					glm::vec4 ttt = invMeshTransMat*glm::vec4(X[3 * iv + 0], X[3 * iv + 1], X[3 * iv + 2], 1.0);
					vv[k] = make_float3(ttt.x, ttt.y, ttt.z);
				}
				float4 bary = GetBarycentricCoordinate(vv[0], vv[1], vv[2], vv[3], vc);
				if (within(bary.x) && within(bary.y) && within(bary.z) && within(bary.w)) {
					vIdx[i] = cubeIdx * 5 + j;
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

void ModelGrid::UpdatePointCoords(float4* v, int n, float4* vOri)
{
	int tet_number = GetTetNumber();
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
		else if (vOri!=0){
			v[i] = vOri[i];
		}
		else{
			std::cerr << "error getting glyph deformed position from mesh" << std::endl; 
		}
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


void ModelGrid::ReinitiateMesh(float3 lensCenter, float lSemiMajorAxis, float lSemiMinorAxis, float3 majorAxis, float focusRatio, float3 lensDir, float4* vOri, int n, Volume* v)
{
	if (!bMeshNeedReinitiation)
		return;

	if (gridType != GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return;

	if (lsgridMesh != 0)
		delete lsgridMesh;
	lsgridMesh = new LineSplitGridMesh<float>(_dmin, _dmax, _n, lensCenter, lSemiMajorAxis, lSemiMinorAxis, majorAxis, focusRatio, lensDir, meshTransMat);

	if (n > 0 && vOri!=0){
		UpdatePointTetId(vOri, n);

		if (useDensityBasedElasticity)
			SetElasticityByTetDensity(n);
		else
			SetElasticitySimple();
	}
	else if (v!=0)
	{
		SetElasticitySimple();
		if (useDensityBasedElasticity)
			SetElasticityByTetDensityOfVolumeCUDA(v);
		else
			SetElasticitySimple();
	}
	else
	{
		SetElasticitySimple();
	}

	lsgridMesh->Initialize(time_step);

	bMeshNeedReinitiation = false;
}



void ModelGrid::SetElasticitySimple()
{
	std::vector<float> density;
	density.resize(lsgridMesh->tet_number);
	for (int i = 0; i < density.size(); i++) {
		density[i] = 500;
	}
	std::copy(&density[0], &density[0] + lsgridMesh->tet_number, lsgridMesh->EL);
}


inline int iDivUp33(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void
d_computeDensityForVolume(cudaExtent volumeSize, float3 spacing, float step, int3 nStep, const float* invMeshTransMat, const int* tet, const float* X, float* density)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
	{
		return;
	}


	float voxelValue = tex3D(volumeTex, x + 0.5, y + 0.5, z + 0.5);
	if (voxelValue < 0.1)
		return;

	//glm::vec4 g_vcTransformed = invMeshTransMat*g_vcOri;
	float4 g_vcTransformed = mat4mulvec4(invMeshTransMat, make_float4(spacing.x*x, spacing.y*y, spacing.z*z, 1.0));


	float3 vc = make_float3(g_vcTransformed.x, g_vcTransformed.y, g_vcTransformed.z);
	float3 tmp = vc / step;
	int3 idx3 = make_int3(floor(tmp.x), floor(tmp.y), floor(tmp.z));

	if (idx3.x < 0 || idx3.y < 0 || idx3.z < 0 || idx3.x >= nStep.x - 1 || idx3.y >= nStep.y - 1 || idx3.z >= nStep.z - 1){
		;
	}
	else{
		int cubeIdx = idx3.x * (nStep.y - 1) * (nStep.z - 1) + idx3.y * (nStep.z - 1) + idx3.z;

		int j = 0;
		for (j = 0; j < 5; j++){
			float3 vv[4];
			int tetId = cubeIdx * 5 + j;
			for (int k = 0; k < 4; k++){
				int iv = tet[tetId * 4 + k];
				vv[k] = make_float3(mat4mulvec4(invMeshTransMat, make_float4(X[3 * iv + 0], X[3 * iv + 1], X[3 * iv + 2],1.0)));
			}
			float4 bary = GetBarycentricCoordinate_device(vv[0], vv[1], vv[2], vv[3], vc);
			if (within_device(bary.x) && within_device(bary.y) && within_device(bary.z) && within_device(bary.w)) {

				atomicAdd(density + (cubeIdx * 5 + j), voxelValue);
				return;
			}
		}
	}

	
}

void ModelGrid::SetElasticityByTetDensityOfVolumeCUDA(Volume* v)
{

	cudaExtent size = v->volumeCuda.size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp33(size.width, blockSize.x), iDivUp33(size.height, blockSize.y), iDivUp33(size.depth, blockSize.z));

	int tet_number = GetTetNumber();

	float* dev_density;
	cudaMalloc((void**)&dev_density, sizeof(float)* tet_number);
	cudaMemset(dev_density, 0, sizeof(float)*tet_number);


	int3 dataSizes = v->size;
	float3 spacing = v->spacing;
	float step = GetStep();
	int3 nStep = GetNumSteps();

	glm::mat4 invMeshTransMat = glm::inverse(meshTransMat);
	float* invMeshTransMatMemPointer = glm::value_ptr(invMeshTransMat);
	float* dev_invMeshTrans;
	cudaMalloc((void**)&dev_invMeshTrans, sizeof(float)* 16);
	cudaMemcpy(dev_invMeshTrans, invMeshTransMatMemPointer, sizeof(float)* 16, cudaMemcpyHostToDevice);

	checkCudaErrors(cudaBindTextureToArray(volumeTex, v->volumeCuda.content, v->volumeCuda.channelDesc));

	d_computeDensityForVolume << <gridSize, blockSize >> >(size, spacing, step, nStep, dev_invMeshTrans, GetTetDev(), GetXDev(), dev_density);


	checkCudaErrors(cudaUnbindTexture(volumeTex));

	
	float* density = lsgridMesh->EL;
	cudaMemcpy(density, dev_density, sizeof(float)*tet_number, cudaMemcpyDeviceToHost);
	
	float* tetVolumeOriginal = lsgridMesh->tetVolumeOriginal;
	float spacingCoeff = spacing.x*spacing.y*spacing.z;
	for (int i = 0; i < tet_number; i++) {
		density[i] = 500 + 100000 * pow(density[i] / (tetVolumeOriginal[i] / spacingCoeff), 2);
	}
	//std::vector<float> forDebug(density, density + tet_number);
}

void ModelGrid::SetElasticityByTetDensity(int n)
{
	int tet_number = GetTetNumber();
	std::vector<int> cnts;
	cnts.resize(tet_number, 0);
	for (int i = 0; i < n; i++){
		int vi = vIdx[i];
		if (vi >= 0 && vi < tet_number){
			cnts[vi]++;
		}
	}
	float* tetVolumeOriginal = lsgridMesh->tetVolumeOriginal;
	std::vector<float> density;
	density.resize(cnts.size());
	//const float base = 400.0f / cnts.size();
	for (int i = 0; i < cnts.size(); i++) {
		//for (int j = 0; j < 5; j++) {
		density[i] = 500 + 800 * pow((float)cnts[i] / tetVolumeOriginal[i], 2);
		//}
	}
	std::copy(&density[0], &density[0] + lsgridMesh->tet_number, lsgridMesh->EL);
	
	//std::vector<float> forDebug(tetVolumeOriginal, tetVolumeOriginal + tet_number);
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

void ModelGrid::Update(float lensCenter[3], float lenDir[3], float lSemiMajorAxis, float lSemiMinorAxis, float focusRatio,float3 majorAxisGlobal)
{
	if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		lsgridMesh->Update(time_step, 1, lensCenter, lenDir, lsgridMesh->meshCenter, lsgridMesh->cutY, lsgridMesh->nStep, lSemiMajorAxis, lSemiMinorAxis, focusRatio, majorAxisGlobal, deformForce);
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

float* ModelGrid::GetXDev()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->dev_X;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->dev_X;
	else
		return NULL;
}

float* ModelGrid::GetXDevOri()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->dev_X_Orig;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->dev_X_Orig;
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

int* ModelGrid::GetTetDev()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->dev_Tet;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->dev_Tet;
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

float3 ModelGrid::GetLensSpaceOrigin()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->gridMin;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->lensSpaceOriginInWorld;
	else
		return  make_float3(0, 0, 0);
}