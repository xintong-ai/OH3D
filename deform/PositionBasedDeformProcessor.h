#ifndef	POSITION_BASED_DEFORM_PROCESSOR_H
#define POSITION_BASED_DEFORM_PROCESSOR_H

#include <thrust/device_vector.h>
#include "Processor.h"

#include <memory>
#include <vector>
#include <ctime>
#include <vector_types.h>
#include <helper_timer.h>
#include "Volume.h"

enum EYE_STATE { inCell, inWall };
enum DATA_DEFORM_STATE { ORIGINAL, DEFORMED};
enum DEFORMED_DATA_TYPE { VOLUME, MESH, PARTICLE };
enum SHAPE_MODEL { CIRCLE, CUBOID };

struct RayCastingParameters;
class Volume;
class VolumeCUDA;
class PolyMesh;
class Particle;
class MatrixManager;
class PositionBasedDeformProcessor :public Processor
{
public:
	SHAPE_MODEL deformModel = CIRCLE;

	std::shared_ptr<Volume> volume = 0;
	std::shared_ptr<PolyMesh> poly = 0;
	std::shared_ptr<Particle> particle = 0;
	
	//the range is usually the range of the data with a little margin
	float3 minPos = make_float3(-3, -3, -3), maxPos = make_float3(70, 120, 165);


	float densityThr = 0.01; //used for volume
	int checkRadius = 1;  //used for volume. can combine with disThr?
	float disThr = 4.1;	//used for poly and particle

	std::shared_ptr<MatrixManager> matrixMgr;

	PositionBasedDeformProcessor(std::shared_ptr<Volume> ori, std::shared_ptr<MatrixManager> _m);
	PositionBasedDeformProcessor(std::shared_ptr<PolyMesh> ori, std::shared_ptr<MatrixManager> _m);
	PositionBasedDeformProcessor(std::shared_ptr<Particle> ori, std::shared_ptr<MatrixManager> _m);

	~PositionBasedDeformProcessor(){
		sdkDeleteTimer(&timer);
		sdkDeleteTimer(&timerFrame);
		if (d_vertexCoords) cudaFree(d_vertexCoords);
		if (d_vertexCoords_init) cudaFree(d_vertexCoords_init);
		if (d_indices) cudaFree(d_indices);
		if (d_numAddedFaces) cudaFree(d_numAddedFaces);
		if (d_vertexDeviateVals) cudaFree(d_vertexDeviateVals);
		if (d_vertexColorVals) cudaFree(d_vertexColorVals);

	};		

	bool isColoringDeformedPart = false;
	
	bool isForceDeform = false;

	bool process(float* modelview, float* projection, int winWidth, int winHeight) override;

	float deformationScale = 5; // for rect, the width of opening
	float deformationScaleVertical = 7; // for rectangular, it is the other side length
	double totalDuration = 3; //time to fully open/close
	double closeDuration = 3;

	void reset(){}

	bool hasOpenAnimeStarted = false;
	bool hasCloseAnimeStarted = false;

	float3 tunnelStart, tunnelEnd;
	float3 rectVerticalDir; // for rectangular, it is the direction of deformationScaleVertical

	float r = 0; //degree of deformation

	bool deformData = true; //sometimes not need to modify the data, but just compute the deformation info like the frame, and just deform the internal data copy

	//for time varying particle dataset
	void updateParticleData(std::shared_ptr<Particle> ori);

	std::shared_ptr<RayCastingParameters> rcp;

private:
	//for time varying particle dataset
	int maxLabel = -1;

	EYE_STATE lastEyeState = inCell;
	DATA_DEFORM_STATE lastDataState = ORIGINAL;
	DEFORMED_DATA_TYPE dataType = VOLUME;

	float* d_vertexCoords = 0;
	float* d_vertexCoords_init = 0;
	unsigned int* d_indices = 0;
	float* d_norms = 0;
	float* d_vertexDeviateVals = 0;
	float* d_vertexColorVals = 0;

	int* d_numAddedFaces = 0;

	void modifyPolyMesh();

	bool processVolumeData(float* modelview, float* projection, int winWidth, int winHeight);
	bool processMeshData(float* modelview, float* projection, int winWidth, int winHeight);
	bool processParticleData(float* modelview, float* projection, int winWidth, int winHeight);

	thrust::device_vector<float4> d_vec_posOrig;
	thrust::device_vector<float4> d_vec_posTarget;

	bool inRange(float3 v);
	void deformDataByDegree(float r);
	void deformDataByDegree2Tunnel(float r, float rClose);
	void resetData();

	std::shared_ptr<VolumeCUDA> volumeCudaIntermediate; //when mixing opening and closing, an intermediate volume is needed

	bool atProperLocationInDeformedData(float3 pos);
	bool atProperLocationInOriData(float3 pos, bool useOriData); //useOriData = true: check if proper in original data; false: check if proper in deformed data

	float lastOpenFinalDegree;
	float3 lastDeformationDirVertical;
	float3 lastTunnelStart, lastTunnelEnd;
	
	void InitCudaSupplies();

	void doVolumeDeform(float degree);
	void doVolumeDeform2Tunnel(float degree, float degreeClose);
	void doPolyDeform(float degree);
	void doParticleDeform(float degree);

	void computeTunnelInfo(float3 centerPoint);

	std::clock_t startOpen;
	std::clock_t startClose;
	float closeStartingRadius;


	float3 targetUpVecInLocal = make_float3(0, 0, 1);	//note! the vector make_float3(0,0,1) may also be used in ImmersiveInteractor class

	StopWatchInterface *timer = 0;
	int fpsCount = 0;
	StopWatchInterface *timerFrame = 0;

};
#endif