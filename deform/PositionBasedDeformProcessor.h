#ifndef	POSITION_BASED_DEFORM_PROCESSOR_H
#define POSITION_BASED_DEFORM_PROCESSOR_H

#include <thrust/device_vector.h>
#include "Processor.h"

#include <memory>
#include <vector>
#include <ctime>
#include <vector_types.h>
#include <helper_timer.h>



enum EYE_STATE { inCell, inWall };
enum DATA_DEFORM_STATE { ORIGINAL, DEFORMED};
enum DEFORMED_DATA_TYPE { VOLUME, MESH, PARTICLE };

class Volume;
class VolumeCUDA;
class PolyMesh;
class MatrixManager;
class PositionBasedDeformProcessor :public Processor
{
public:
	std::shared_ptr<Volume> volume;
	std::shared_ptr<PolyMesh> poly;
	
	std::shared_ptr<Volume> channelVolume;
	std::shared_ptr<MatrixManager> matrixMgr;

	PositionBasedDeformProcessor(std::shared_ptr<Volume> ori, std::shared_ptr<MatrixManager> _m, std::shared_ptr<Volume> ch);
	PositionBasedDeformProcessor(std::shared_ptr<PolyMesh> ori, std::shared_ptr<MatrixManager> _m, std::shared_ptr<Volume> ch);
	
	~PositionBasedDeformProcessor(){
		sdkDeleteTimer(&timer);
		sdkDeleteTimer(&timerFrame);

		if (d_vertexCoords) cudaFree(d_vertexCoords);
		if (d_vertexCoords_init) cudaFree(d_vertexCoords_init);
		if (d_indices) cudaFree(d_indices);
		if (d_faceValid) cudaFree(d_faceValid);
		if (d_numAddedFaces) cudaFree(d_numAddedFaces);
	};		


	bool process(float* modelview, float* projection, int winWidth, int winHeight) override;

	EYE_STATE lastEyeState = inCell;
	DATA_DEFORM_STATE lastDataState = ORIGINAL;
	DEFORMED_DATA_TYPE dataType = VOLUME;

	float deformationScale = 5; // for rect, the width of opening
	float deformationScaleVertical = 7; // for rectangular, it is the other side length

	void reset(){}

	bool hasOpenAnimeStarted = false;
	bool hasCloseAnimeStarted = false;

	float3 tunnelStart, tunnelEnd;
	float3 rectVerticalDir; // for rectangular, it is the direction of deformationScaleVertical

private:
	float* d_vertexCoords = 0;
	float* d_vertexCoords_init = 0;
	unsigned int* d_indices = 0;
	float* d_norms = 0;

	bool* d_faceValid = 0;
	int* d_numAddedFaces = 0;

	void modifyPolyMesh();

	bool processVolumeData(float* modelview, float* projection, int winWidth, int winHeight);
	bool processMeshData(float* modelview, float* projection, int winWidth, int winHeight);
	
	std::shared_ptr<VolumeCUDA> volumeCudaIntermediate; //when mixing opening and closing, an intermediate volume is needed

	float3 spacing;
	bool inDeformedCell(float3 pos);

	float lastOpenFinalDegree;
	float3 lastDeformationDirVertical;
	float3 lastTunnelStart, lastTunnelEnd;


	void InitCudaSupplies();

	void doVolumeDeform(float degree);
	void doVolumeDeform2Tunnel(float degree, float degreeClose);
	void doChannelVolumeDeform();
	void doPolyDeform(float degree);

	void computeTunnelInfo(float3 centerPoint);

	std::clock_t startOpen;
	std::clock_t startClose;
	double totalDuration = 3;

	float closeStartingRadius;
	double closeDuration = 3;

	float3 targetUpVecInLocal = make_float3(0, 0, 1);	//note! the vector make_float3(0,0,1) may also be used in ImmersiveInteractor class

	StopWatchInterface *timer = 0;
	int fpsCount = 0;
	StopWatchInterface *timerFrame = 0;

};
#endif