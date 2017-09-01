#ifndef	POSITION_BASED_DEFORM_PROCESSOR_FOR_TV_H
#define POSITION_BASED_DEFORM_PROCESSOR_FOR_TV_H

#include <thrust/device_vector.h>
#include "Processor.h"

#include <memory>
#include <vector>
#include <ctime>
#include <vector_types.h>
#include <helper_timer.h>


class Volume;
class VolumeCUDA;

class Particle;
class MatrixManager;
class PolyMesh;

class PositionBasedDeformProcessor;
class PolyRenderable;

class PositionBasedDeformProcessorForTV :public Processor
{
public:
	bool justChangedForRenderer = false;

	//std::shared_ptr<Particle> particle;
	//std::shared_ptr<MatrixManager> matrixMgr;
	std::shared_ptr<PolyMesh> polyMesh;

	std::vector < std::shared_ptr<PolyMesh>> polyMeshes;


	std::vector < std::shared_ptr<PolyMesh>> polyMeshesOri; //the renderable and the processor can both operate and change polyMeshes. Save the original copy here. It may not save all info, but only saves the one that might be changed.
	void saveOriginalCopyOfMeshes();


	int timeStart = 6, timeEnd = 7;
	int curT = -1;
	int numInter = 80;
	std::vector<std::vector<int>> cellMaps;//given the index of a region in last timestep. get the index of the region in next timestep with the same label
	//std::vector<std::vector<int>> labelToIdMap; //given the label of a region in the current timestep, find the index of 
	int maxLabel = -1;

	std::shared_ptr<PositionBasedDeformProcessor> positionBasedDeformProcessor = 0;

	void turnActive();

	//PositionBasedDeformProcessorForTV(std::shared_ptr<Particle> ori, std::shared_ptr<MatrixManager> _m, std::shared_ptr<Volume> ch);
	PositionBasedDeformProcessorForTV(){};

	~PositionBasedDeformProcessorForTV(){
		sdkDeleteTimer(&timer);
		sdkDeleteTimer(&timerFrame);
	};

	//bool isColoringDeformedPart = false;

	//bool isForceDeform = false;

	bool process(float* modelview, float* projection, int winWidth, int winHeight) override;


	//for time varying particle dataset
	void resetPolyMeshes();
private:

	StopWatchInterface *timer = 0;
	int fpsCount = 0;
	StopWatchInterface *timerFrame = 0;

	void copyPolyMesh(int meshId);

};
#endif