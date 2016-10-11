#ifndef LINESPLIT_MODEL_GRID_H
#define LINESPLIT_MODEL_GRID_H
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <ctime>

#include <thrust/device_vector.h>

enum GRID_TYPE{
	UNIFORM_GRID,
	LINESPLIT_UNIFORM_GRID
};

template <class TYPE>
class GridMesh;
template <class TYPE>
class LineSplitGridMesh;
class Volume;
class Lens;
class LineLens3D;
class Particle;

class LineSplitModelGrid
{
	std::vector<Lens*> *lenses = 0;

	GridMesh<float>* gridMesh;
	LineSplitGridMesh<float>* lsgridMesh;
	bool bMeshNeedReinitiation = false;
	glm::mat4 meshTransMat;

	std::vector<float4> vBaryCoord;
	std::vector<int> vIdx;
	
	const float	time_step = 1 / 30.0;
	float deformForce = 0;// 30;

	clock_t startTime;

	//density related
	void SetElasticitySimple(float v);
	void SetElasticityByTetDensityOfPartice(int n); //suppose the tet id for particles have been well set
	void SetElasticityByTetDensityOfVolumeCUDA(std::shared_ptr<Volume> v);

	//currently stored
	float dataMin[3], dataMax[3];

	//for both mesh


	//for line mesh only
	void InitPointTetId_LineSplitMesh(float4* v, int n);

	//for uniform mesh only
	void InitPointTetId_UniformMesh(float4* v, int n);


	//only needed for particle
	thrust::device_vector<float4> d_vec_vOri;
	thrust::device_vector<int> d_vec_vIdx;
	thrust::device_vector<float4> d_vec_vBaryCoord;
	thrust::device_vector<float4> d_vec_v;
	thrust::device_vector<float> d_vec_brightness;
	thrust::device_vector<char> d_vec_feature;


public:
	GRID_TYPE gridType = LINESPLIT_UNIFORM_GRID;
	
	int meshResolution;

	void SetLenses(std::vector<Lens*> *_lenses){ lenses = _lenses; }

	//density related
	int elasticityMode = 1;

	void SetElasticityForParticle(std::shared_ptr<Particle> p);
	void SetElasticityForVolume(std::shared_ptr<Volume> v);
	float minElas = 0, maxElasEstimate = 1; //used for draw the mesh in image
	void UpdateMeshDevElasticity(); //need more work to finish

	//for both mesh
	LineSplitModelGrid(float dmin[3], float dmax[3], int n);
	bool ProcessParticleDeformation(float* modelview, float* projection, int winWidth, int winHeight, std::shared_ptr<Particle> particle, float* glyphSizeScale, float* glyphBright = 0, bool isFreezingFeature = false, int snappedGlyphId = -1, int snappedFeatureId = -1);


	void initThrustVectors(std::shared_ptr<Particle>); //only needed for particle

	//for circle mesh only
	void InitializeUniformGrid(std::shared_ptr<Particle> p); //the info of gridMesh only need to be initialized once, so use a different initail stretagy with lsgridMesh

	void UpdateUniformMesh(float* _mv);
	void UpdatePointCoordsAndBright_UniformMesh(std::shared_ptr<Particle> p, float* brightness,  float* _mv);

	//for line mesh only
	void setReinitiationNeed(){ bMeshNeedReinitiation = true; }
	void ReinitiateMeshForParticle(LineLens3D* l, std::shared_ptr<Particle> p);
	void ReinitiateMeshForVolume(LineLens3D * l, std::shared_ptr<Volume> v);	
	void UpdateMesh(float3 lensCenter, float3 lensDir, float lSemiMajorAxis, float lSemiMinorAxis, float focusRatio, float3 majorAxisGlobal);
	void UpdatePointCoordsAndBright_LineMeshLens_Thrust(std::shared_ptr<Particle> p, float* brightness, LineLens3D * l, bool isFreezingFeature, int snappedFeatureId);
	void MoveMesh(float3 moveDir);
	bool ProcessVolumeDeformation(float* modelview, float* projection, int winWidth, int winHeight, std::shared_ptr<Volume> volume);


	//currently for line mesh only
	void setDeformForce(float f){ deformForce = f; }
	float getDeformForce(){ return deformForce; }

	//mesh region attributes;
	float3 GetZDiretion();
	float3 GetXDiretion(); 
	float3 GetLensSpaceOrigin();

	//mesh attributes
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

};
#endif