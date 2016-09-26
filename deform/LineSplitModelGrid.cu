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
#include <helper_timer.h>

texture<float, 3, cudaReadModeElementType>  volumeTex;

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

		float alpha = 0.25f;//for FPM data
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
				//thrust::get<4>(t) = 1.0f;
				thrust::get<4>(t) = max(1.0f / (alpha * abs(abs(lensCen2PMinorProj) - lSemiMinorAxis / focusRatio) + 1.0f), dark);
			}
		}
		else{
			if (abs(lensCen2PMinorProj) < lSemiMinorAxis / focusRatio){
				thrust::get<4>(t) = max(1.0f / (alpha * abs(abs(lensCen2PMajorProj) - lSemiMajorAxis) + 1.0f), dark);		
			}
			else{
				//thrust::get<4>(t) = 1.0f;
				float dMaj = abs(lensCen2PMajorProj) - lSemiMajorAxis;
				float dMin = abs(lensCen2PMinorProj) - lSemiMinorAxis / focusRatio;
				thrust::get<4>(t) = max(1.0f / (alpha * sqrt(dMin*dMin + dMaj*dMaj) + 1.0f), dark);
			}
			//thrust::get<4>(t) = 1.0f;
		}
	}
	functor_UpdatePointCoordsAndBrightByLineLensMesh(thrust::device_ptr<int> _dev_ptr_tet, thrust::device_ptr<float> _dev_ptr_X, int _tet_number, float3 _lensCenter, float _lSemiMajorAxisGlobal, float _lSemiMinorAxisGlobal, float3 _majorAxisGlobal, float _focusRatio, float3 _lensDir, bool _isFreezingFeature, int _snappedFeatureId) : dev_ptr_tet(_dev_ptr_tet), dev_ptr_X(_dev_ptr_X), tet_number(_tet_number), lensCenter(_lensCenter), lSemiMajorAxis(_lSemiMajorAxisGlobal), lSemiMinorAxis(_lSemiMinorAxisGlobal), majorAxis(_majorAxisGlobal), focusRatio(_focusRatio), lensDir(_lensDir), isFreezingFeature(_isFreezingFeature), snappedFeatureId(_snappedFeatureId){}
};


inline int iDivUp33(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}


__global__ void
d_computeTranferDensityForVolume(cudaExtent volumeSize, float3 spacing, float step, int3 nStep, const float* invMeshTransMat, const int* tet, const float* X, float* density, int* count,  int densityTransferMode)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int z = blockIdx.z*blockDim.z + threadIdx.z;

	if (x >= volumeSize.width || y >= volumeSize.height || z >= volumeSize.depth)
	{
		return;
	}


	float voxelValue = tex3D(volumeTex, x + 0.5, y + 0.5, z + 0.5);
	float usedValue;
	if (densityTransferMode == 1){
		usedValue = voxelValue;
	}
	else if(densityTransferMode == 2){
		//key values:
		// < 0.1945: ourside backgronud
		// 110-140 / 0.214-0.272 : from cortex to ventrical
		// 155-250 / 0.302-0.486 : cortex and skull
		//250+ /0.486: ventrical

		if (voxelValue < 0.2)
			usedValue = (0.2 - voxelValue) / 0.2;
		else if (voxelValue < 0.5)
			voxelValue = 0;
		else
			usedValue = (voxelValue - 0.5) /0.5;
	}
	else if(densityTransferMode == 3){
		float4 grad = make_float4(0.0);

		int indz1 = z - 2, indz2 = z + 2;
		if (indz1 < 0)	indz1 = 0;
		if (indz2 > volumeSize.depth - 1) indz2 = volumeSize.depth - 1;
		grad.z = (tex3D(volumeTex, x + 0.5, y + 0.5, indz2 + 0.5) - tex3D(volumeTex, x + 0.5, y + 0.5, indz1 + 0.5)) / (indz2 - indz1);

		int indy1 = y - 2, indy2 = y + 2;
		if (indy1 < 0)	indy1 = 0;
		if (indy2 > y >= volumeSize.height - 1) indy2 = y >= volumeSize.height - 1;
		grad.y = (tex3D(volumeTex, x + 0.5, indy2 + 0.5, z + 0.5) - tex3D(volumeTex, x + 0.5, indy1 + 0.5, z + 0.5)) / (indy2 - indy1);

		int indx1 = x - 2, indx2 = x + 2;
		if (indx1 < 0)	indx1 = 0;
		if (indx2 > volumeSize.width - 1) indx2 = volumeSize.width - 1;
		grad.x = (tex3D(volumeTex, indx2 + 0.5, y + 0.5, z + 0.5) - tex3D(volumeTex, indx1 + 0.5, y + 0.5, z + 0.5)) / (indx2 - indx1);
		usedValue = length(grad);
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
			float4 bary = GetBarycentricCoordinate2(vv[0], vv[1], vv[2], vv[3], vc);
			if (within_device(bary.x) && within_device(bary.y) && within_device(bary.z) && within_device(bary.w)) {
				atomicAdd(count + (cubeIdx * 5 + j), 1);
				atomicAdd(density + (cubeIdx * 5 + j), usedValue);
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

	//lsgridMesh = new LineSplitGridMesh<float>(dmin, dmax, n);
	lsgridMesh = new LineSplitGridMesh<float>();

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
				float4 bary = GetBarycentricCoordinate2(vv[0], vv[1], vv[2], vv[3], vc);
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

void LineSplitModelGrid::InitPointTetId_UniformMesh(float4* v, int n)
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
		if (idx3.x < 0 || idx3.y < 0 || idx3.z < 0 || idx3.x >= nStep.x - 1 || idx3.y >= nStep.y - 1 || idx3.z >= nStep.z - 1){
			vIdx[i] = -1;
		}
		else{
			int idx = idx3.x * (nStep.y - 1) * (nStep.z - 1)
				+ idx3.y * (nStep.z - 1) + idx3.z;
			int j;
			for (j = 0; j < 5; j++){
				float3 vv[4];
				int tetId = idx * 5 + j;
				for (int k = 0; k < 4; k++){
					int iv = tet[tetId * 4 + k];
					vv[k] = make_float3(X[3 * iv + 0], X[3 * iv + 1], X[3 * iv + 2]);
				}
				float4 bary = GetBarycentricCoordinate2(vv[0], vv[1], vv[2], vv[3], vc);
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
		if (vi == -1){
			v[i] = make_float4(-100, -100, -100, 1);
		}
		else{
			float4 vb = vBaryCoord[i];
			float3 vv[4];
			for (int k = 0; k < 4; k++){
				int iv = tet[vi * 4 + k];
				vv[k] = make_float3(X[3 * iv + 0], X[3 * iv + 1], X[3 * iv + 2]);
			}
			v[i] = make_float4(vb.x * vv[0] + vb.y * vv[1] + vb.z * vv[2] + vb.w * vv[3], 1);
		}
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

	InitPointTetId_UniformMesh(&(p->posOrig[0]), p->numParticles);
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

	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	//can add a lsgridMesh clean function before Initialize. or else delete and readd the lsgridMesh
	if (lsgridMesh != 0)
		delete lsgridMesh;
	lsgridMesh = new LineSplitGridMesh<float>(dataMin, dataMax, meshResolution, l->c, l->lSemiMajorAxisGlobal, l->lSemiMinorAxisGlobal, l->majorAxisGlobal, l->focusRatio, l->lensDir, meshTransMat);

	InitPointTetId_LineSplitMesh(&(p->posOrig[0]), p->numParticles);

	SetElasticityForParticle(p);

	lsgridMesh->Initialize(time_step);

	timer->stop();
	std::cout << "time used to construct mesh: " << sdkGetAverageTimerValue(&timer) /1000.f << std::endl;
	sdkDeleteTimer(&timer);

	bMeshNeedReinitiation = false;
}


void LineSplitModelGrid::SetElasticityForParticle(std::shared_ptr<Particle> p)
{
	if (elasticityMode == 1)
		SetElasticityByTetDensityOfPartice(p->numParticles);
	else if (elasticityMode == 0)
		SetElasticitySimple(200);
	else{
		std::cerr << "density mode error" << std::endl;
		exit(0);
	}
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
	{
		std::cerr << "error ReinitiateMeshForVolume" << std::endl;
		exit(0);
	}

	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	if (lsgridMesh != 0)
		delete lsgridMesh;
	lsgridMesh = new LineSplitGridMesh<float>(dataMin, dataMax, meshResolution, l->c, l->lSemiMajorAxisGlobal, l->lSemiMinorAxisGlobal, l->majorAxisGlobal, l->focusRatio, l->lensDir, meshTransMat);

	SetElasticityForVolume(v);
	
	lsgridMesh->Initialize(time_step);

	timer->stop();
	std::cout << "time used to construct mesh: " << sdkGetAverageTimerValue(&timer) / 1000.f << std::endl;
	sdkDeleteTimer(&timer);

	bMeshNeedReinitiation = false;
}





void computeDensityForVolumeCPU(cudaExtent volumeSize, float3 spacing, float step, int3 nStep, const float* invMeshTransMat, const int* tet, const float* X, float* density, int* count, int densityTransferMode, Volume *v)
{
	for (int z = 0; z < volumeSize.depth; z++){
		for (int y = 0; y < volumeSize.height; y++){
			for (int x = 0; x < volumeSize.width; x++){
				int ind = z*volumeSize.width*volumeSize.height + y*volumeSize.width + x;
				float voxelValue = v->values[ind];
				float usedValue;
				if (densityTransferMode == 1){
					usedValue = voxelValue;
				}
				else if (densityTransferMode == 2){
					//key values:
					// < 0.1945: ourside backgronud
					// 110-140 / 0.214-0.272 : from cortex to ventrical
					// 155-250 / 0.302-0.486 : cortex and skull
					//250+ /0.486: ventrical

					if (voxelValue < 0.2)
						usedValue = (0.2 - voxelValue) / 0.2;
					else if (voxelValue < 0.5)
						voxelValue = 0;
					else
						usedValue = (voxelValue - 0.5) / 0.5;
				}
				else if (densityTransferMode == 3){
					float4 grad = make_float4(0.0);

					int indz1 = z - 2, indz2 = z + 2;
					if (indz1 < 0)	indz1 = 0;
					if (indz2 > volumeSize.depth - 1) indz2 = volumeSize.depth - 1;
					grad.z = (v->values[indz2*volumeSize.width*volumeSize.height + y*volumeSize.width + x] - v->values[indz1*volumeSize.width*volumeSize.height + y*volumeSize.width + x]) / (indz2 - indz1);

					int indy1 = y - 2, indy2 = y + 2;
					if (indy1 < 0)	indy1 = 0;
					if (indy2 > y >= volumeSize.height - 1) indy2 = y >= volumeSize.height - 1;
					grad.y = (v->values[z*volumeSize.width*volumeSize.height + indy2*volumeSize.width + x] - v->values[z*volumeSize.width*volumeSize.height + indy1*volumeSize.width + x]) / (indy2 - indy1);

					int indx1 = x - 2, indx2 = x + 2;
					if (indx1 < 0)	indx1 = 0;
					if (indx2 > volumeSize.width - 1) indx2 = volumeSize.width - 1;
					grad.x = (v->values[z*volumeSize.width*volumeSize.height + y*volumeSize.width + indx2] - v->values[z*volumeSize.width*volumeSize.height + y*volumeSize.width + indx1]) / (indx2 - indx1);
					usedValue = length(grad);
				}

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
						float4 bary = GetBarycentricCoordinate2(vv[0], vv[1], vv[2], vv[3], vc);
						if (within(bary.x) && within(bary.y) && within(bary.z) && within(bary.w)) {
							count[(cubeIdx * 5 + j)] = count[(cubeIdx * 5 + j)] + 1;
							density[(cubeIdx * 5 + j)] = density[(cubeIdx * 5 + j)] + usedValue;
						}
					}
				}

			}
		}
	}
	return;
}



void LineSplitModelGrid::SetElasticityByTetDensityOfVolumeCUDA(std::shared_ptr<Volume> v)
{

	cudaExtent size = v->volumeCuda.size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp33(size.width, blockSize.x), iDivUp33(size.height, blockSize.y), iDivUp33(size.depth, blockSize.z));

	int tet_number = GetTetNumber();

	float* dev_density;
	int* dev_count;
	cudaMalloc((void**)&dev_density, sizeof(float)* tet_number);
	cudaMemset(dev_density, 0, sizeof(float)*tet_number);
	cudaMalloc((void**)&dev_count, sizeof(int)* tet_number);
	cudaMemset(dev_count, 0, sizeof(int)*tet_number);

	int3 dataSizes = v->size;
	float3 spacing = v->spacing;
	float step = GetStep();
	int3 nStep = GetNumSteps();

	glm::mat4 invMeshTransMat = glm::inverse(meshTransMat);
	float* invMeshTransMatMemPointer = glm::value_ptr(invMeshTransMat);
	float* dev_invMeshTrans;
	cudaMalloc((void**)&dev_invMeshTrans, sizeof(float)* 16);
	cudaMemcpy(dev_invMeshTrans, invMeshTransMatMemPointer, sizeof(float)* 16, cudaMemcpyHostToDevice);


	//checkCudaErrors(cudaBindTextureToArray(volumeTex, v->volumeCuda.content, v->volumeCuda.channelDesc));
	//d_computeTranferDensityForVolume << <gridSize, blockSize >> >(size, spacing, step, nStep, dev_invMeshTrans, GetTetDev(), GetXDev(), dev_density, dev_count, elasticityMode);
	//checkCudaErrors(cudaUnbindTexture(volumeTex));
	//float* density = lsgridMesh->EL;
	//cudaMemcpy(density, dev_density, sizeof(float)*tet_number, cudaMemcpyDeviceToHost);
	//int* count = new int[tet_number];
	//cudaMemcpy(count, dev_count, sizeof(int)*tet_number, cudaMemcpyDeviceToHost);

	float* density = lsgridMesh->EL;
	int* count = new int[tet_number];
	memset(density, 0, sizeof(float)*tet_number);
	memset(count, 0, sizeof(int)*tet_number);
	computeDensityForVolumeCPU(size, spacing, step, nStep, invMeshTransMatMemPointer, GetTet(), GetX(), density, count, elasticityMode, v.get());

	//std::vector<float> forDebug2(density, density + tet_number);
	//std::vector<float> forDebug3(count, count + tet_number);




	if (elasticityMode == 3)
	{
		for (int i = 0; i < tet_number; i++) {
			if (count[i] == 0){
				density[i] = 100;
			}
			else{
				density[i] = 100 + 2000000 * pow(density[i] / count[i], 2);
			}
		}
	}
	else{
		for (int i = 0; i < tet_number; i++) {
			if (count[i] == 0){
				density[i] = 100;
			}
			else{
				density[i] = 100 + 20000 * pow(density[i] / count[i], 2);
			}
		}
	}
	minElas = 100;
	maxElasEstimate = 2100;
	
	//std::vector<float> forDebug(density, density + tet_number);

	delete count;
	cudaFree(dev_density);
	cudaFree(dev_count);
}




void LineSplitModelGrid::SetElasticityForVolume(std::shared_ptr<Volume> v)
{
	if (elasticityMode == 1 || elasticityMode == 2 || elasticityMode == 3)
		SetElasticityByTetDensityOfVolumeCUDA(v);
	else if (elasticityMode == 0)
		SetElasticitySimple(200);
	else{
		std::cerr << "density mode error" << std::endl;
		exit(0);
	}
}

void LineSplitModelGrid::UpdateMeshDevElasticity()
{
	lsgridMesh->UpdateMeshDevElasticity();
}

void LineSplitModelGrid::UpdateMesh(float lensCenter[3], float lenDir[3], float lSemiMajorAxis, float lSemiMinorAxis, float focusRatio, float3 majorAxisGlobal)
{
	if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		lsgridMesh->UpdateLineMesh(time_step, 4, lensCenter, lenDir, lsgridMesh->meshCenter, lsgridMesh->cutY, lsgridMesh->nStep, lSemiMajorAxis, lSemiMinorAxis, focusRatio, majorAxisGlobal, deformForce);
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