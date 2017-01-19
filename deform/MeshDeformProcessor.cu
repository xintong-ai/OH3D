#include "MeshDeformProcessor.h"
#include "GridMesh.h"
#include "LineSplitGridMesh.h"

#include "myDefine.h"
#include "Volume.h"
#include "Particle.h"
#include "Lens.h"
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
#include "TransformFunc.h"
#include <helper_timer.h>

texture<float, 3, cudaReadModeElementType>  volumeTex;

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

MeshDeformProcessor::MeshDeformProcessor(float dmin[3], float dmax[3], int n)
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

void MeshDeformProcessor::setParticleData(std::shared_ptr<Particle> _p)
{
	particle = _p;
	int n = particle->numParticles;
	d_vec_vOri.resize(n);
	thrust::copy(particle->posOrig.begin(), particle->posOrig.end(), d_vec_vOri.begin());
	d_vec_vIdx.resize(n);
	d_vec_vBaryCoord.resize(n);
	d_vec_v.resize(n);
	d_vec_brightness.resize(n);
	d_vec_feature.resize(n);
	if (particle->hasFeature){
		thrust::copy(particle->feature.begin(), particle->feature.end(), d_vec_feature.begin());
	}
}

void MeshDeformProcessor::setVolumeData(std::shared_ptr<Volume> _v)
{
	volume = _v;
}

void MeshDeformProcessor::InitPointTetId_LineSplitMesh(float4* v, int n)
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

void MeshDeformProcessor::InitPointTetId_UniformMesh(float4* v, int n)
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

void MeshDeformProcessor::InitializeUniformGrid(std::shared_ptr<Particle> p)
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

void MeshDeformProcessor::ReinitiateMeshForParticle(LineLens3D * l, std::shared_ptr<Particle> p)
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
	lsgridMesh = new LineSplitGridMesh<float>(length(l->frontBaseCenter - l->c), -length(l->estMeshBottomCenter - l->c), meshResolution, l->c, l->lSemiMajorAxisGlobal, l->lSemiMinorAxisGlobal / l->focusRatio, l->majorAxisGlobal, l->lensDir, meshTransMat);

	InitPointTetId_LineSplitMesh(&(p->posOrig[0]), p->numParticles);

	SetElasticityForParticle(p);

	lsgridMesh->Initialize(time_step);

	timer->stop();
	std::cout << "time used to construct mesh: " << sdkGetAverageTimerValue(&timer) /1000.f << std::endl;
	sdkDeleteTimer(&timer);

	bMeshNeedReinitiation = false;
}


void MeshDeformProcessor::SetElasticityForParticle(std::shared_ptr<Particle> p)
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

void MeshDeformProcessor::SetElasticityByTetDensityOfPartice(int n)
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

void MeshDeformProcessor::SetElasticitySimple(float v)
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


void MeshDeformProcessor::ReinitiateMeshForVolume(LineLens3D * l, std::shared_ptr<Volume> v)
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
	lsgridMesh = new LineSplitGridMesh<float>(length(l->frontBaseCenter - l->c), -length(l->estMeshBottomCenter - l->c), meshResolution, l->c, l->lSemiMajorAxisGlobal, l->lSemiMinorAxisGlobal / l->focusRatio, l->majorAxisGlobal, l->lensDir, meshTransMat);

	SetElasticityForVolume(v);
	
	lsgridMesh->Initialize(time_step);

	timer->stop();
	std::cout << "time used to construct mesh: " << sdkGetAverageTimerValue(&timer) / 1000.f << std::endl;
	sdkDeleteTimer(&timer);

	bMeshNeedReinitiation = false;
}


bool MeshDeformProcessor::ProcessVolumeDeformation(float* modelview, float* projection, int winWidth, int winHeight, std::shared_ptr<Volume> volume)
{
	if (lenses == NULL || lenses->size() == 0)
		return false;
	Lens *l = lenses->back();

	if (l->type != TYPE_LINE || l->isConstructing)
		return false;

	if (l->justChanged){
		setReinitiationNeed();

		float3 dmin, dmax;
		volume->GetPosRange(dmin, dmax);

		((LineLens3D*)l)->UpdateObjectLineLens(winWidth, winHeight, modelview, projection, dmin, dmax); //need to be placed to a better place!
		l->justChanged = false;
	}

	//besides the lens change, the mesh may also need to reinitiate from other commands
	ReinitiateMeshForVolume((LineLens3D*)l, volume);

	UpdateLineSplitMesh(((LineLens3D*)l)->c, ((LineLens3D*)l)->lensDir, ((LineLens3D*)l)->lSemiMajorAxisGlobal, ((LineLens3D*)l)->lSemiMinorAxisGlobal, ((LineLens3D*)l)->focusRatio, ((LineLens3D*)l)->majorAxisGlobal);

	return true;
};

bool MeshDeformProcessor::ProcessParticleDeformation(float* modelview, float* projection, int winWidth, int winHeight, std::shared_ptr<Particle> particle)
{
	if (lenses == NULL || lenses->size() == 0)
		return false;
	Lens *l = lenses->back();

	float* glyphSizeScale = &(particle->glyphSizeScale[0]);
	float* glyphBright = &(particle->glyphBright[0]);
	bool isFreezingFeature = particle->isFreezingFeature;
	int snappedGlyphId = particle->snappedGlyphId;
	int snappedFeatureId = particle->snappedFeatureId;

	if (l->type == TYPE_LINE){
		if (l->isConstructing){
			return false;
		}

		if (l->justChanged){
			startTime = clock();
			setReinitiationNeed();
			l->justChanged = false;
		}
		
		ReinitiateMeshForParticle((LineLens3D*)l, particle);

		double secondsPassed = (clock() - startTime) / CLOCKS_PER_SEC;
		//if (secondsPassed > 15)
			//return true;

		UpdateLineSplitMesh(((LineLens3D*)l)->c, ((LineLens3D*)l)->lensDir, ((LineLens3D*)l)->lSemiMajorAxisGlobal, ((LineLens3D*)l)->lSemiMinorAxisGlobal, ((LineLens3D*)l)->focusRatio, ((LineLens3D*)l)->majorAxisGlobal);
	}
	else if (l->type == TYPE_CIRCLE){
		UpdateUniformMesh(modelview);
	}
}

bool MeshDeformProcessor::process(float* modelview, float* projection, int winWidth, int winHeight)
{
	if (!isActive)
		return false;

	if (data_type == USE_PARTICLE){
		return ProcessParticleDeformation(modelview, projection, winWidth, winHeight, particle);
	}
	else if (data_type == USE_VOLUME){
		return ProcessVolumeDeformation(modelview, projection, winWidth, winHeight, volume);
	}
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



void MeshDeformProcessor::SetElasticityByTetDensityOfVolumeCUDA(std::shared_ptr<Volume> v)
{

	cudaExtent size = v->volumeCuda.size;
	unsigned int dim = 32;
	dim3 blockSize(dim, dim, 1);
	dim3 gridSize(iDivUp(size.width, blockSize.x), iDivUp(size.height, blockSize.y), iDivUp(size.depth, blockSize.z));

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


	checkCudaErrors(cudaBindTextureToArray(volumeTex, v->volumeCuda.content, v->volumeCuda.channelDesc));
	d_computeTranferDensityForVolume << <gridSize, blockSize >> >(size, spacing, step, nStep, dev_invMeshTrans, GetTetDev(), GetXDev(), dev_density, dev_count, elasticityMode);
	checkCudaErrors(cudaUnbindTexture(volumeTex));
	float* density = lsgridMesh->EL;
	cudaMemcpy(density, dev_density, sizeof(float)*tet_number, cudaMemcpyDeviceToHost);
	int* count = new int[tet_number];
	cudaMemcpy(count, dev_count, sizeof(int)*tet_number, cudaMemcpyDeviceToHost);

	//float* density = lsgridMesh->EL;
	//int* count = new int[tet_number];
	//memset(density, 0, sizeof(float)*tet_number);
	//memset(count, 0, sizeof(int)*tet_number);
	//computeDensityForVolumeCPU(size, spacing, step, nStep, invMeshTransMatMemPointer, GetTet(), GetX(), density, count, elasticityMode, v.get());

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




void MeshDeformProcessor::SetElasticityForVolume(std::shared_ptr<Volume> v)
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

void MeshDeformProcessor::UpdateMeshDevElasticity()
{
	lsgridMesh->UpdateMeshDevElasticity();
}

void MeshDeformProcessor::UpdateLineSplitMesh(float3 lensCenter, float3 lenDir, float lSemiMajorAxis, float lSemiMinorAxis, float focusRatio, float3 majorAxisGlobal)
{
	lsgridMesh->UpdateLineMesh(time_step, 4, lensCenter, lenDir, lsgridMesh->cutY, lsgridMesh->nStep, lSemiMajorAxis, lSemiMinorAxis, focusRatio, majorAxisGlobal, deformForce);
	meshJustDeformed = true;
	return;
}

void MeshDeformProcessor::UpdateUniformMesh(float* _mv)
{
	Lens* l = lenses->back();
	float3 lensCen = l->c;
	float focusRatio = l->focusRatio;
	float radius = ((CircleLens3D*)l)->objectRadius;
	float _invmv[16];
	invertMatrix(_mv, _invmv);
	float3 cameraObj = make_float3(Camera2Object(make_float4(0, 0, 0, 1), _invmv));
	float3 lensDir = normalize(cameraObj - lensCen);

	gridMesh->Update(time_step, 64, lensCen, lensDir, focusRatio, radius);
	meshJustDeformed = true;
}

/////////////////////////////////////// attributes getters /////////////////////

int MeshDeformProcessor::GetTNumber()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->t_number;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->t_number;
	else
		return -1;
}

int* MeshDeformProcessor::GetT()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->T;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->T;
	else
		return NULL;
}

float* MeshDeformProcessor::GetX()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->X;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->X;
	else
		return NULL;
}

float* MeshDeformProcessor::GetXDev()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->dev_X;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->dev_X;
	else
		return NULL;
}

float* MeshDeformProcessor::GetXDevOri()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->dev_X_Orig;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->dev_X_Orig;
	else
		return NULL;
}


int MeshDeformProcessor::GetNumber()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->number;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->number;
	else
		return 0;
}

unsigned int* MeshDeformProcessor::GetL()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->L;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->L;
	else
		return NULL;
}

float* MeshDeformProcessor::GetE()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->EL;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->EL;
	else
		return NULL;
}

int MeshDeformProcessor::GetLNumber()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->l_number;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->l_number;
	else
		return 0;
}

float3 MeshDeformProcessor::GetGridMin()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->gridMin;
	else {
		std::cerr << "error GetGridMin()" << std::endl;
	}
}

float3 MeshDeformProcessor::GetGridMax()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->gridMax;
	else {
		std::cerr << "error GetGridMax()" << std::endl;
	}
}

int3 MeshDeformProcessor::GetNumSteps()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return make_int3(gridMesh->nStep[0], gridMesh->nStep[1], gridMesh->nStep[2]);
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return make_int3(lsgridMesh->nStep[0], lsgridMesh->nStep[1], lsgridMesh->nStep[2]);
	else
		return make_int3(0, 0, 0);
}

float MeshDeformProcessor::GetStep()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->step;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->step;
	else
		return 0;
}
int* MeshDeformProcessor::GetTet()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->Tet;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->Tet;
	else
		return NULL;
}

int* MeshDeformProcessor::GetTetDev()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->dev_Tet;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->dev_Tet;
	else
		return NULL;
}

int MeshDeformProcessor::GetTetNumber()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->tet_number;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->tet_number;
	else
		return NULL;
}


float3 MeshDeformProcessor::GetZDiretion()
{
	Lens *l = lenses->back();
	 if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		 return ((LineLens3D*)l)->lensDir;
	 else{
		 return  make_float3(0, 0, 0);
	 }
}

float3 MeshDeformProcessor::GetXDiretion()
{
	Lens *l = lenses->back();
	if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return ((LineLens3D*)l)->majorAxisGlobal;
	else{
		return  make_float3(0, 0, 0);
	}
}


float3 MeshDeformProcessor::GetLensSpaceOrigin()
{
	if (gridType == GRID_TYPE::UNIFORM_GRID)
		return gridMesh->gridMin;
	else if (gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID)
		return lsgridMesh->lensSpaceOriginInWorld;
	else
		return  make_float3(0, 0, 0);
}