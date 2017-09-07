#ifndef	POSITION_BASED_DEFORM_PROCESSOR_H
#define POSITION_BASED_DEFORM_PROCESSOR_H

#include <thrust/device_vector.h>
#include "Processor.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <ctime>
#include <vector_types.h>
#include <helper_timer.h>
#include "Volume.h"

enum SYSTEM_STATE { ORIGINAL, DEFORMED, OPENING, CLOSING, MIXING };
enum DEFORMED_DATA_TYPE { VOLUME, MESH, PARTICLE };
enum SHAPE_MODEL { CIRCLE, CUBOID, PHYSICALLY };


class TunnelTimer
{
public:
	TunnelTimer()
	{
		sdkCreateTimer(&timer);
	}
	void init(float ot, float it = 0)
	{
		outTime = ot;
		initTime = it;
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
	}
	bool out(){ return initTime >= 0 && (timer->getTime() + initTime > outTime); }
	bool inUse() { return initTime >= 0; }
	void end(){ sdkStopTimer(&timer); initTime = -1; }
	float getTime(){ return fmin(outTime, timer->getTime() + initTime); }
	~TunnelTimer()
	{
		sdkDeleteTimer(&timer);
	}
private:
	float outTime;
	float initTime = -1; //initTime >=0 means timer has been inited
	StopWatchInterface *timer;
};


struct RayCastingParameters;
class Volume;
class VolumeCUDA;
class PolyMesh;
class Particle;
class MatrixManager;
class PositionBasedDeformProcessor :public Processor
{
public:

	std::shared_ptr<Volume> volume = 0;
	std::shared_ptr<RayCastingParameters> rcp;

	std::shared_ptr<PolyMesh> poly = 0;
	std::shared_ptr<Particle> particle = 0;
	
	//the range is usually the range of the data with a little margin
	float3 minPos = make_float3(-3, -3, -3), maxPos = make_float3(70, 120, 165);


	float densityThr = 0.01; //used for volume
	int checkRadius = 1;  //used for volume. can combine with disThr?
	float disThr = 4.1;	//used for poly and particle


	PositionBasedDeformProcessor(std::shared_ptr<Volume> ori, std::shared_ptr<MatrixManager> _m);
	PositionBasedDeformProcessor(std::shared_ptr<PolyMesh> ori, std::shared_ptr<MatrixManager> _m);
	PositionBasedDeformProcessor(std::shared_ptr<Particle> ori, std::shared_ptr<MatrixManager> _m);



	bool isColoringDeformedPart = false;
	
	bool isForceDeform = false;


	bool setOutTime(float v){
		if (lastDataState == ORIGINAL) { outTime = v * 1000; return true; }
		else{ return false; }
	}

	//shape model
	SHAPE_MODEL getShapeModel(){ return shapeModel; }
	bool setShapeModel(SHAPE_MODEL s){
		if (lastDataState == ORIGINAL) { shapeModel = s; return true; }
		else{ return false; }
	}

	
	//tunnel parameter
	//cuboid
	float getDeformationScaleVertical(){ return deformationScaleVertical; }
	float getDeformationScale(){ return deformationScale; }
	bool setDeformationScale(float v){
		if (lastDataState == ORIGINAL) { deformationScale = v; return true; }
		else{ return false; }
	}
	bool setDeformationScaleVertical(float v){
		if (lastDataState == ORIGINAL) { deformationScaleVertical = v; return true; }
		else{ return false; }
	}


	void reset(){}

	float3 getTunnelStart(){ return tunnelStart; }
	float3 getTunnelEnd(){ return tunnelEnd; }
	float3 getRectVerticalDir(){ return rectVerticalDir; }

	float r = 0; //degree of deformation

	bool deformData = true; //sometimes not need to modify the data, but just compute the deformation info like the frame, and just deform the internal data copy

	//for time varying particle dataset
	void updateParticleData(std::shared_ptr<Particle> ori);
	void polyMeshDataUpdated();

	~PositionBasedDeformProcessor(){
		sdkDeleteTimer(&timer);
		sdkDeleteTimer(&timerFrame);
		if (d_vertexCoords) cudaFree(d_vertexCoords);
		if (d_norms) cudaFree(d_norms);
		if (d_vertexCoords_init) cudaFree(d_vertexCoords_init);
		if (d_indices) cudaFree(d_indices);
		if (d_numAddedFaces) cudaFree(d_numAddedFaces);
		if (d_vertexDeviateVals) cudaFree(d_vertexDeviateVals);
		if (d_vertexColorVals) cudaFree(d_vertexColorVals);

	};
	bool process(float* modelview, float* projection, int winWidth, int winHeight) override;

private:
	SYSTEM_STATE lastDataState = ORIGINAL;
	DEFORMED_DATA_TYPE dataType = VOLUME;
	SHAPE_MODEL shapeModel = CUBOID;

	//tunnel functions
	void computeTunnelInfo(float3 centerPoint);
	bool sameTunnel();
	//tunnel info
	float3 lastDeformationDirVertical;
	float3 lastTunnelStart, lastTunnelEnd;
	float3 tunnelStart, tunnelEnd;
	float3 rectVerticalDir; // for rectangular, it is the direction of deformationScaleVertical
	void storeCurrentTunnel(){
		lastTunnelStart = tunnelStart;
		lastTunnelEnd = tunnelEnd;
		lastDeformationDirVertical = rectVerticalDir;
	}
	//circle
	float radius = 10;

	float deformationScale = 5; // for rect, the width of opening
	float deformationScaleVertical = 7; // for rectangular, it is the other side length



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

	
	void deformDataByDegree(float r);
	void deformDataByDegree2Tunnel(float r, float rClose);


	std::shared_ptr<VolumeCUDA> volumeCudaIntermediate; //when mixing opening and closing, an intermediate volume is needed
	
	bool inRange(float3 v); 
	void resetData();
	bool atProperLocationInDeformedData(float3 pos);
	bool atProperLocation(float3 pos, bool useOriData); //useOriData = true: check if proper in original data; false: check if proper in deformed data (with a in-tunnel check at the beginning)


	
	void InitCudaSupplies();

	void doVolumeDeform(float degree);
	void doVolumeDeform2Tunnel(float degree, float degreeClose);
	void doPolyDeform(float degree);
	void doParticleDeform(float degree);



	std::shared_ptr<MatrixManager> matrixMgr;
	float3 targetUpVecInLocal = make_float3(0, 0, 1);	//note! the vector make_float3(0,0,1) may also be used in ImmersiveInteractor class. They are supposed to be the same

	StopWatchInterface *timer = 0;
	int fpsCount = 0;
	StopWatchInterface *timerFrame = 0;

	TunnelTimer tunnelTimer1, tunnelTimer2;
	float outTime = 3000;//miliseconds
};
#endif