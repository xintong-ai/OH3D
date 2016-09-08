#ifndef LINESPLIT_MODEL_GRID_H
#define LINESPLIT_MODEL_GRID_H
#include <vector>
#include <ModelGrid.h> //plan to inheritate ModelGrid class in the future

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <thrust/device_vector.h>


template <class TYPE>
class GridMesh;
template <class TYPE>
class LineSplitGridMesh;
class Volume;
class LineLens3D;
class Particle;

class LineSplitModelGrid
{
	GridMesh<float>* gridMesh;
	LineSplitGridMesh<float>* lsgridMesh;
	bool bMeshNeedReinitiation = false;
	glm::mat4 meshTransMat;

	std::vector<float4> vBaryCoord;
	std::vector<int> vIdx;
	const float	time_step = 1 / 30.0;
	float deformForce = 32;


	void SetElasticitySimple(float v);
	void SetElasticityByTetDensityOfPartice(int n); //suppose the tet id for particles have been well set
	void SetElasticityByTetDensityOfVolumeCUDA(std::shared_ptr<Volume> v);
	void SetElasticityByTetVarianceOfVolumeCUDA(std::shared_ptr<Volume> v);

	//currently stored
	int _n;
	float _dmin[3], _dmax[3];

	//only needed for particle
	thrust::device_vector<float4> d_vec_vOri;
	thrust::device_vector<int> d_vec_vIdx;
	thrust::device_vector<float4> d_vec_vBaryCoord;
	thrust::device_vector<float4> d_vec_v;
	thrust::device_vector<float> d_vec_brightness;
	thrust::device_vector<char> d_vec_feature;

public:
	GRID_TYPE gridType = LINESPLIT_UNIFORM_GRID;
	bool useDensityBasedElasticity = true;


	void SetElasticityForParticle(std::shared_ptr<Particle> p);
	void SetElasticityForVolume(std::shared_ptr<Volume> v);
	void UpdateMeshDevElasticity();


	LineSplitModelGrid(float dmin[3], float dmax[3], int n);
	void setDeformForce(float f){ deformForce = f; }
	float getDeformForce(){ return deformForce; }

	void LineSplitModelGrid::UpdatePointCoordsAndBright_LineMeshLens_Thrust(Particle * p, float* brightness, LineLens3D * l, bool isFreezingFeature, int snappedFeatureId);

	void initThrustVectors(std::shared_ptr<Particle>); //only needed for particle
	
	void UpdatePointTetId(float4* v, int n);

	void ReinitiateMeshForParticle(LineLens3D* l, std::shared_ptr<Particle> p);
	void ReinitiateMeshForVolume(LineLens3D * l, std::shared_ptr<Volume> v);

	void setReinitiationNeed(){ bMeshNeedReinitiation = true; }

	void Initialize(float time_step);
	void UpdateMesh(float lensCenter[3], float lenDir[3], float lSemiMajorAxis, float lSemiMinorAxis, float focusRatio, float3 majorAxisGlobal);
	void MoveMesh(float3 moveDir);

	float minElas = 0, maxElasEstimate = 1; //used for draw the mesh in image


	int GetTNumber();
	int* GetT();
	float* GetX();
	float* GetXDev();
	float* GetXDevOri();

	int GetNumber();
	unsigned int* GetL();
	int GetLNumber();
	float3 GetGridMin();
	float3 GetGridMax();
	int3 GetNumSteps();
	float GetStep();
	float* GetE();
	int* GetTet();
	int* GetTetDev();
	int GetTetNumber();
	float3 GetLensSpaceOrigin();

};
#endif