#ifndef MODEL_GRID_H
#define MODEL_GRID_H
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <thrust/device_vector.h>


template <class TYPE>
class GridMesh;
template <class TYPE>
class LineSplitGridMesh;
class Volume;
enum GRID_TYPE{
	UNIFORM_GRID,
	LINESPLIT_UNIFORM_GRID
};
class ModelGrid
{
	GridMesh<float>* gridMesh;
	LineSplitGridMesh<float>* lsgridMesh;
	bool bMeshNeedReinitiation = false;
	glm::mat4 meshTransMat;

	std::vector<float4> vBaryCoord;
	std::vector<int> vIdx;
	const float	time_step = 1 / 30.0;
	float deformForce = 32;


	void SetElasticity(float* v);
	void SetElasticitySimple();
	void SetElasticityByTetDensity(int n); //suppose the tet id for particles have been well set
	void SetElasticityByTetDensityOfVolumeCUDA(Volume* v);

	//currently stored
	int _n;
	float _dmin[3], _dmax[3];


	

public:
	GRID_TYPE gridType = UNIFORM_GRID;
	bool useDensityBasedElasticity = true;

	ModelGrid(float dmin[3], float dmax[3], int n);
	ModelGrid(float dmin[3], float dmax[3], int n, bool useLineSplitGridMesh);
	void setDeformForce(float f){ deformForce = f; }
	float getDeformForce(){ return deformForce; }

	void UpdatePointCoords(float4* v, int n, float4* vOri = 0);
	void UpdatePointCoords_LineMeshLens_Thrust(float4* v, int n);
	void UpdatePointCoordsAndBright_LineMeshLens_Thrust(float4* v, float* brightness, int n, float3 lensCenter, float lSemiMajorAxisGlobal, float lSemiMinorAxisGlobal, float3 majorAxisGlobal, float focusRatio, float3 lensDir, bool isFreezingFeature, int snappedFeatureId, char *feature = 0);
	//these thrust variables should be placed in a better place
	thrust::device_vector<float4> d_vec_vOri;
	thrust::device_vector<int> d_vec_vIdx;
	thrust::device_vector<float4> d_vec_vBaryCoord;
	thrust::device_vector<float4> d_vec_v;
	thrust::device_vector<float> d_vec_brightness;
	thrust::device_vector<char> d_vec_feature;


	void InitGridDensity(float4* v, int n);
	void UpdatePointTetId(float4* v, int n);

	void ReinitiateMesh(float3 lensCenter, float lSemiMajorAxis, float lSemiMinorAxis, float3 majorAxis, float focusRatio, float3 lensDir, float4* vOri, int n, Volume* v=0);

	void setReinitiationNeed(){ bMeshNeedReinitiation = true; }
	
	void Initialize(float time_step);
	void Update(float lensCenter[3], float lenDir[3], float focusRatio, float radius);
	void Update(float lensCenter[3], float lenDir[3], float lSemiMajorAxis, float lSemiMinorAxis, float focusRatio, float3 majorAxisGlobal);

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