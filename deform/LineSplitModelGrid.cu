#include <LineSplitModelGrid.h>
#include <GridMesh.h>
#include <LineSplitGridMesh.h>

#include <defines.h>
#include <Volume.h>
#include <Particle.h>
#include <Lens.h>
#include <thrust/execution_policy.h>
#include <thrust/uninitialized_copy.h>

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


struct functor_UpdatePointCoordsAndBrightByLineLensMesh
{
	thrust::device_ptr<int> dev_ptr_tet;;
	thrust::device_ptr<float> dev_ptr_X;
	int tet_number;
	float3 lensCenter;
	float lSemiMajorAxis;
	float lSemiMinorAxis;
	float3 majorAxis;
	float focusRatio;
	float3 lensDir;
	bool isFreezingFeature;
	int snappedFeatureId;
	template<typename Tuple>
	__device__ __host__ void operator() (Tuple t)
	{
		if (isFreezingFeature && snappedFeatureId == thrust::get<5>(t)){
			thrust::get<3>(t) = thrust::get<0>(t);
			thrust::get<4>(t) = 1.0;
			return;
		}
		
		int vi = thrust::get<1>(t);
		if (vi >= 0 && vi < tet_number){
			float4 vb = thrust::get<2>(t);
			float3 vv[4];
			for (int k = 0; k < 4; k++){
				int iv = dev_ptr_tet[vi * 4 + k];
				vv[k] = make_float3(dev_ptr_X[3 * iv + 0], dev_ptr_X[3 * iv + 1], dev_ptr_X[3 * iv + 2]);
			}
			thrust::get<3>(t) = make_float4(vb.x * vv[0] + vb.y * vv[1] + vb.z * vv[2] + vb.w * vv[3], 1);
		}
		else{
			thrust::get<3>(t) = thrust::get<0>(t);
		}

		thrust::get<4>(t) = 1.0;
		const float dark = 0.5;
		float3 lenCen2P = make_float3(thrust::get<0>(t)) - lensCenter;
		//float lensCen2PProj = dot(lenCen2P, lensDir);	
		//if (lensCen2PProj < 0){
		//	float lensCen2PMajorProj = dot(lenCen2P, majorAxis);
		//	if (abs(lensCen2PMajorProj) < lSemiMajorAxis){
		//		float3 minorAxis = cross(lensDir, majorAxis);
		//		float lensCen2PMinorProj = dot(lenCen2P, minorAxis);
		//		if (abs(lensCen2PMinorProj) < lSemiMinorAxis / focusRatio){
		//			float candLight = 1.0f / (0.5f * abs(lensCen2PProj) + 1.0f);
		//			thrust::get<4>(t) = candLight>dark ? candLight : dark;
		//		}
		//	}
		//}
		float alpha = 0.25f;
		float lensCen2PMajorProj = dot(lenCen2P, majorAxis);
		float3 minorAxis = cross(lensDir, majorAxis);
		float lensCen2PMinorProj = dot(lenCen2P, minorAxis);
		if (abs(lensCen2PMajorProj) < lSemiMajorAxis){
			if (abs(lensCen2PMinorProj) < lSemiMinorAxis / focusRatio){
				float lensCen2PProj = dot(lenCen2P, lensDir);
				if (lensCen2PProj < 0){		
					//float candLight = 1.0f / (alpha * abs(lensCen2PProj) + 1.0f);
					//thrust::get<4>(t) = candLight>dark ? candLight : dark;
					thrust::get<4>(t) = max(1.0f / (alpha * abs(lensCen2PProj) + 1.0f), dark);
				}
			}
			else{
				thrust::get<4>(t) = dark;
			}
		}
		else{
			thrust::get<4>(t) = dark;
		}
	}
	functor_UpdatePointCoordsAndBrightByLineLensMesh(thrust::device_ptr<int> _dev_ptr_tet, thrust::device_ptr<float> _dev_ptr_X, int _tet_number, float3 _lensCenter, float _lSemiMajorAxisGlobal, float _lSemiMinorAxisGlobal, float3 _majorAxisGlobal, float _focusRatio, float3 _lensDir, bool _isFreezingFeature, int _snappedFeatureId) : dev_ptr_tet(_dev_ptr_tet), dev_ptr_X(_dev_ptr_X), tet_number(_tet_number), lensCenter(_lensCenter), lSemiMajorAxis(_lSemiMajorAxisGlobal), lSemiMinorAxis(_lSemiMinorAxisGlobal), majorAxis(_majorAxisGlobal), focusRatio(_focusRatio), lensDir(_lensDir), isFreezingFeature(_isFreezingFeature), snappedFeatureId(_snappedFeatureId){}
};


inline int iDivUp33(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}


__global__ void
d_computeTranferDensityForVolume(cudaExtent volumeSize, float3 spacing, float step, int3 nStep, const float* invMeshTransMat, const int* tet, const float* X, float* density, float v1, float v2, int densityTransferMode)
//densityTransferMode==0: low value for input between v1 and v2
//densityTransferMode==1: high value for input between v1 and v2
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
	{
		return;
	}


	float voxelValue = tex3D(volumeTex, x + 0.5, y + 0.5, z + 0.5);
	if (densityTransferMode == 0){
		if (voxelValue > v1 && voxelValue < v2)
			voxelValue = 0;
		else if (voxelValue <= v1)
			voxelValue = 1 - voxelValue / (v1 + 0.00000001); //1.00000001 to avoid setting v0==0
		else
			voxelValue = (voxelValue - v2) / (1.00000001 - v2);
	}
	else if (densityTransferMode == 1){
		if (voxelValue < v1)
			voxelValue = (voxelValue / (v1 + 0.00000001))/5;
		else if (voxelValue < v2)
			voxelValue = 1;
		else
			voxelValue = (1 - (voxelValue - v2) / (1.00000001 - v2))/5;//1.00000001 to avoid setting v2==1
	}
	else if (densityTransferMode == 2){
		if (voxelValue > 0.6)
			voxelValue = voxelValue;
		else
			voxelValue = 0;
	}


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
				vv[k] = make_float3(mat4mulvec4(invMeshTransMat, make_float4(X[3 * iv + 0], X[3 * iv + 1], X[3 * iv + 2], 1.0)));
			}
			float4 bary = GetBarycentricCoordinate(vv[0], vv[1], vv[2], vv[3], vc);
			if (within_device(bary.x) && within_device(bary.y) && within_device(bary.z) && within_device(bary.w)) {

				atomicAdd(density + (cubeIdx * 5 + j), voxelValue);
				return;
			}
		}
	}


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
				vv[k] = make_float3(mat4mulvec4(invMeshTransMat, make_float4(X[3 * iv + 0], X[3 * iv + 1], X[3 * iv + 2], 1.0)));
			}
			float4 bary = GetBarycentricCoordinate(vv[0], vv[1], vv[2], vv[3], vc);
			if (within_device(bary.x) && within_device(bary.y) && within_device(bary.z) && within_device(bary.w)) {

				atomicAdd(density + (cubeIdx * 5 + j), voxelValue);
				return;
			}
		}
	}


}


__global__ void
d_computeVarianceForVolume(cudaExtent volumeSize, float3 spacing, float step, int3 nStep, const float* invMeshTransMat, const int* tet, const float* X, float* density, float* tetVolumeOriginal, float* variance)
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
				vv[k] = make_float3(mat4mulvec4(invMeshTransMat, make_float4(X[3 * iv + 0], X[3 * iv + 1], X[3 * iv + 2], 1.0)));
			}
			float4 bary = GetBarycentricCoordinate(vv[0], vv[1], vv[2], vv[3], vc);
			if (within_device(bary.x) && within_device(bary.y) && within_device(bary.z) && within_device(bary.w)) {
				float mean = *(density + tetId) / (*(tetVolumeOriginal + tetId));
				atomicAdd(variance + tetId, (voxelValue - mean)*(voxelValue - mean));
				return;
			}
		}
	}


}


LineSplitModelGrid::LineSplitModelGrid(float dmin[3], float dmax[3], int n)
{
	meshResolution = n;
	dataMin[0] = dmin[0];
	dataMin[1] = dmin[1];
	dataMin[2] = dmin[2];
	dataMax[0] = dmax[0];
	dataMax[1] = dmax[1];
	dataMax[2] = dmax[2];

	lsgridMesh = new LineSplitGridMesh<float>(dmin, dmax, n);

	volumeTex.normalized = false;
	volumeTex.filterMode = cudaFilterModeLinear;
	volumeTex.addressMode[0] = cudaAddressModeBorder;
	volumeTex.addressMode[1] = cudaAddressModeBorder;
	volumeTex.addressMode[2] = cudaAddressModeBorder;

	gridMesh = new GridMesh<float>(dmin, dmax, n);
}


void LineSplitModelGrid::initThrustVectors(std::shared_ptr<Particle> p)
{
	int n = p->numParticles;
	d_vec_vOri.resize(n);
	thrust::copy(p->posOrig.begin(), p->posOrig.end(), d_vec_vOri.begin());
	d_vec_vIdx.resize(n);
	d_vec_vBaryCoord.resize(n);
	d_vec_v.resize(n);
	d_vec_brightness.resize(n);
	d_vec_feature.resize(n);
	if (p->hasFeature){
		thrust::copy(p->feature.begin(), p->feature.end(), d_vec_feature.begin());
	}
}


void LineSplitModelGrid::InitPointTetId_LineSplitMesh(float4* v, int n)
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

	thrust::copy(&vIdx[0], &vIdx[0] + n, d_vec_vIdx.begin());
	thrust::copy(&vBaryCoord[0], &vBaryCoord[0] + n, d_vec_vBaryCoord.begin());
}



void LineSplitModelGrid::InitGridDensity_UniformMesh(float4* v, int n)
{
	float3 gridMin = GetGridMin();
	float3 gridMax = GetGridMax();
	int3 nStep = GetNumSteps();
	float step = GetStep();
	int* tet = GetTet();
	float* X = GetX();
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
				break;
			}
		}
	}

}




void LineSplitModelGrid::UpdatePointCoordsAndBright_LineMeshLens_Thrust(Particle * p, float* brightness, LineLens3D * l, bool isFreezingFeature, int snappedFeatureId)
{
	if (isFreezingFeature)
	{
		if (!p->hasFeature)
		{
			std::cout << "error of feature in particle data" << std::endl;
			exit(0);
		}
	}

	thrust::device_ptr<int> dev_ptr_tet(GetTetDev());
	thrust::device_ptr<float> dev_ptr_X(GetXDev());
	int tet_number = GetTetNumber();

	thrust::for_each(
		thrust::make_zip_iterator(
		thrust::make_tuple(
		d_vec_vOri.begin(),
		d_vec_vIdx.begin(),
		d_vec_vBaryCoord.begin(),
		d_vec_v.begin(),
		d_vec_brightness.begin(),
		d_vec_feature.begin()
		)),
		thrust::make_zip_iterator(
		thrust::make_tuple(
		d_vec_vOri.end(),
		d_vec_vIdx.end(),
		d_vec_vBaryCoord.end(),
		d_vec_v.end(),
		d_vec_brightness.end(),
		d_vec_feature.end()
		)),
		functor_UpdatePointCoordsAndBrightByLineLensMesh(dev_ptr_tet, dev_ptr_X, tet_number, l->c, l->lSemiMajorAxisGlobal, l->lSemiMinorAxisGlobal, l->majorAxisGlobal, l->focusRatio, l->lensDir, isFreezingFeature, snappedFeatureId));

	thrust::copy(d_vec_v.begin(), d_vec_v.end(), &(p->pos[0]));
	thrust::copy(d_vec_brightness.begin(), d_vec_brightness.end(), brightness);

}

void LineSplitModelGrid::UpdatePointCoordsUniformMesh(float4* v, int n)
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

void LineSplitModelGrid::InitializeUniformGrid(std::shared_ptr<Particle> p)
{
	if (gridType != GRID_TYPE::UNIFORM_GRID)
	{
		std::cerr << "error InitializeUniformGrid" << std::endl;
		exit(0);
	}

	//can add a gridMesh clean function before Initialize. or else delete and readd the gridMesh
	if (gridMesh != 0)
		delete gridMesh;
	gridMesh = new GridMesh<float>(dataMin, dataMax, meshResolution);

	InitGridDensity_UniformMesh(&(p->pos[0]), p->numParticles);
	SetElasticityForParticle(p);
	gridMesh->Initialize(time_step);
}

void LineSplitModelGrid::ReinitiateMeshForParticle(LineLens3D * l, std::shared_ptr<Particle> p)
{
	if (!bMeshNeedReinitiation)
		return;

	if (gridType != GRID_TYPE::LINESPLIT_UNIFORM_GRID)
	{
		std::cerr << "error ReinitiateMeshForParticle" << std::endl;
		exit(0);
	}

	//can add a lsgridMesh clean function before Initialize. or else delete and readd the lsgridMesh
	if (lsgridMesh != 0)
		delete lsgridMesh;
	lsgridMesh = new LineSplitGridMesh<float>(dataMin, dataMax, meshResolution, l->c, l->lSemiMajorAxisGlobal, l->lSemiMinorAxisGlobal, l->majorAxisGlobal, l->focusRatio, l->lensDir, meshTransMat);

	InitPointTetId_LineSplitMesh(&(p->posOrig[0]), p->numParticles);

	SetElasticityForParticle(p);

	lsgridMesh->Initialize(time_step);

	bMeshNeedReinitiation = false;
}


void LineSplitModelGrid::SetElasticityForParticle(std::shared_ptr<Particle> p)
{
	if (useDensityBasedElasticity)
		SetElasticityByTetDensityOfPartice(p->numParticles);
	else
		SetElasticitySimple(200);
}

void LineSplitModelGrid::SetElasticityByTetDensityOfPartice(int n)
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
	std::vector<float> density;
	density.resize(cnts.size());
	if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID){
		float* tetVolumeOriginal = lsgridMesh->tetVolumeOriginal;
		for (int i = 0; i < cnts.size(); i++) {
			density[i] = 500 + 1000 * pow((float)cnts[i] / tetVolumeOriginal[i], 2);
		}
	}
	else{
		for (int i = 0; i < cnts.size(); i++) {
			density[i] = 500 + 1000 * pow((float)cnts[i], 2);
		}
	}
	minElas = 500;
	maxElasEstimate = 1500;
	std::copy(&density[0], &density[0] + tet_number, GetE());

	//std::vector<float> forDebug(tetVolumeOriginal, tetVolumeOriginal + tet_number);
}



void LineSplitModelGrid::SetElasticitySimple(float v)
{
	std::vector<float> density;
	density.resize(GetTetNumber());
	for (int i = 0; i < density.size(); i++) {
		density[i] = v;
	}
	std::copy(&density[0], &density[0] + GetTetNumber(), GetE());
	minElas = v-1;
	maxElasEstimate = v+1;
}


void LineSplitModelGrid::ReinitiateMeshForVolume(LineLens3D * l, std::shared_ptr<Volume> v)
{
	if (!bMeshNeedReinitiation)
		return;

	if (gridType != GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return;

	if (lsgridMesh != 0)
		delete lsgridMesh;
	lsgridMesh = new LineSplitGridMesh<float>(dataMin, dataMax, meshResolution, l->c, l->lSemiMajorAxisGlobal, l->lSemiMinorAxisGlobal, l->majorAxisGlobal, l->focusRatio, l->lensDir, meshTransMat);


	SetElasticityForVolume(v);
	
	lsgridMesh->Initialize(time_step);

	bMeshNeedReinitiation = false;
}

void LineSplitModelGrid::SetElasticityByTetDensityOfVolumeCUDA(std::shared_ptr<Volume> v)
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

	//d_computeDensityForVolume << <gridSize, blockSize >> >(size, spacing, step, nStep, dev_invMeshTrans, GetTetDev(), GetXDev(), dev_density);
	d_computeTranferDensityForVolume << <gridSize, blockSize >> >(size, spacing, step, nStep, dev_invMeshTrans, GetTetDev(), GetXDev(), dev_density, 0.2, 0.6, 1);//MGHT1
	//d_computeTranferDensityForVolume << <gridSize, blockSize >> >(size, spacing, step, nStep, dev_invMeshTrans, GetTetDev(), GetXDev(), dev_density, -1, 0.3, 2);//nek256

	checkCudaErrors(cudaUnbindTexture(volumeTex));


	float* density = lsgridMesh->EL;
	cudaMemcpy(density, dev_density, sizeof(float)*tet_number, cudaMemcpyDeviceToHost);

	float* tetVolumeOriginal = lsgridMesh->tetVolumeOriginal;
	float spacingCoeff = spacing.x*spacing.y*spacing.z;
	//for (int i = 0; i < tet_number; i++) {
	//	density[i] = 100 + 2000 * pow(density[i] / (tetVolumeOriginal[i] / spacingCoeff), 3);
	//}
	for (int i = 0; i < tet_number; i++) {
		float dd = density[i] / (tetVolumeOriginal[i] / spacingCoeff);
		if (dd < 0.5) dd = 0;
		density[i] = 100 + 2000 * pow(dd, 2);
	}
	minElas = 100;
	maxElasEstimate = 2100;
	//std::vector<float> forDebug(density, density + tet_number);
}


void LineSplitModelGrid::SetElasticityByTetVarianceOfVolumeCUDA(std::shared_ptr<Volume> v)
{

	cudaExtent size = v->volumeCuda.size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp33(size.width, blockSize.x), iDivUp33(size.height, blockSize.y), iDivUp33(size.depth, blockSize.z));

	int tet_number = GetTetNumber();

	float* dev_density, *dev_variance;
	cudaMalloc((void**)&dev_density, sizeof(float)* tet_number);
	cudaMemset(dev_density, 0, sizeof(float)*tet_number);
	cudaMalloc((void**)&dev_variance, sizeof(float)* tet_number);
	cudaMemset(dev_variance, 0, sizeof(float)*tet_number);

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
	d_computeVarianceForVolume << <gridSize, blockSize >> >(size, spacing, step, nStep, dev_invMeshTrans, GetTetDev(), GetXDev(), dev_density, lsgridMesh->dev_tetVolumeOriginal, dev_variance);


	checkCudaErrors(cudaUnbindTexture(volumeTex));


	float* variance = lsgridMesh->EL;
	cudaMemcpy(variance, dev_variance, sizeof(float)*tet_number, cudaMemcpyDeviceToHost);

	float* tetVolumeOriginal = lsgridMesh->tetVolumeOriginal;
	float spacingCoeff = spacing.x*spacing.y*spacing.z;
	for (int i = 0; i < tet_number; i++) {
		variance[i] = 500 + 100000 * (variance[i] / (tetVolumeOriginal[i] / spacingCoeff));
	}
	std::vector<float> forDebug(variance, variance + tet_number);
}


void LineSplitModelGrid::SetElasticityForVolume(std::shared_ptr<Volume> v)
{
	if (useDensityBasedElasticity)
		SetElasticityByTetDensityOfVolumeCUDA(v);
	//SetElasticityByTetVarianceOfVolumeCUDA(v);
	else
		SetElasticitySimple(200);
}

void LineSplitModelGrid::UpdateMeshDevElasticity()
{
	lsgridMesh->UpdateMeshDevElasticity();
}

void LineSplitModelGrid::UpdateMesh(float lensCenter[3], float lenDir[3], float lSemiMajorAxis, float lSemiMinorAxis, float focusRatio, float3 majorAxisGlobal)
{
	if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		lsgridMesh->Update(time_step, 4, lensCenter, lenDir, lsgridMesh->meshCenter, lsgridMesh->cutY, lsgridMesh->nStep, lSemiMajorAxis, lSemiMinorAxis, focusRatio, majorAxisGlobal, deformForce);
	return;
}

void LineSplitModelGrid::UpdateUniformMesh(float lensCenter[3], float lenDir[3], float focusRatio, float radius)
{
	gridMesh->Update(time_step, 64, lensCenter, lenDir, focusRatio, radius);
}

void LineSplitModelGrid::MoveMesh(float3 moveDir)
{
	lsgridMesh->MoveMesh(moveDir);
}

/////////////////////////////////////// attributes getters /////////////////////

int LineSplitModelGrid::GetTNumber()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->t_number;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->t_number;
	else
		return -1;
}

int* LineSplitModelGrid::GetT()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->T;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->T;
	else
		return NULL;
}

float* LineSplitModelGrid::GetX()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->X;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->X;
	else
		return NULL;
}

float* LineSplitModelGrid::GetXDev()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->dev_X;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->dev_X;
	else
		return NULL;
}

float* LineSplitModelGrid::GetXDevOri()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->dev_X_Orig;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->dev_X_Orig;
	else
		return NULL;
}


int LineSplitModelGrid::GetNumber()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->number;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->number;
	else
		return 0;
}

unsigned int* LineSplitModelGrid::GetL()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->L;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->L;
	else
		return NULL;
}

float* LineSplitModelGrid::GetE()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->EL;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->EL;
	else
		return NULL;
}

int LineSplitModelGrid::GetLNumber()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->l_number;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->l_number;
	else
		return 0;
}

float3 LineSplitModelGrid::GetGridMin()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->gridMin;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->gridMin;
	else
		return make_float3(0, 0, 0);
}

float3 LineSplitModelGrid::GetGridMax()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->gridMax;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->gridMax;
	else
		return make_float3(0, 0, 0);
}

int3 LineSplitModelGrid::GetNumSteps()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return make_int3(gridMesh->nStep[0], gridMesh->nStep[1], gridMesh->nStep[2]);
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return make_int3(lsgridMesh->nStep[0], lsgridMesh->nStep[1], lsgridMesh->nStep[2]);
	else
		return make_int3(0, 0, 0);
}

float LineSplitModelGrid::GetStep()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->step;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->step;
	else
		return 0;
}
int* LineSplitModelGrid::GetTet()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->Tet;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->Tet;
	else
		return NULL;
}

int* LineSplitModelGrid::GetTetDev()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->dev_Tet;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->dev_Tet;
	else
		return NULL;
}

int LineSplitModelGrid::GetTetNumber()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->tet_number;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->tet_number;
	else
		return NULL;
}

float3 LineSplitModelGrid::GetLensSpaceOrigin()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->gridMin;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->lensSpaceOriginInWorld;
	else
		return  make_float3(0, 0, 0);
}