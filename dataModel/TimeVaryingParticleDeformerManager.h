#ifndef	POSITION_BASED_DEFORM_PROCESSOR_FOR_TV_H
#define POSITION_BASED_DEFORM_PROCESSOR_FOR_TV_H

#include <thrust/device_vector.h>
#include "Processor.h"

#include <memory>
#include <vector>
#include <ctime>
#include <vector_types.h>
#include <helper_timer.h>


class Particle;
class MatrixManager;
class PolyMesh;

class PositionBasedDeformProcessor;
class PolyRenderable;

class TimeVaryingParticleDeformerManager :public Processor
{
public:
	bool justChangedForRenderer = false;

	std::shared_ptr<PolyMesh> polyMesh;
	std::vector < std::shared_ptr<PolyMesh>> polyMeshes;


	std::vector < std::shared_ptr<PolyMesh>> polyMeshesOri; //the renderable and the processor can both operate and change polyMeshes. Save the original copy here. It may not save all info, but only saves the one that might be changed.


	int timeStart = 6, timeEnd = 7;
	int curT = -1;
	int numInter = 80;
	std::vector<std::vector<int>> cellMaps;//given the index of a region in last timestep. get the index of the region in next timestep with the same label


	std::shared_ptr<PositionBasedDeformProcessor> positionBasedDeformProcessor = 0;

	void turnActive();

	TimeVaryingParticleDeformerManager(){};
	~TimeVaryingParticleDeformerManager(){
		sdkDeleteTimer(&timer);
		sdkDeleteTimer(&timerFrame);
	};

	bool process(float* modelview, float* projection, int winWidth, int winHeight) override;

	void finishedMeshesSetting(){ saveOriginalCopyOfMeshes(); };
	void resetPolyMeshes();
private:

	StopWatchInterface *timer = 0;
	int fpsCount = 0;
	StopWatchInterface *timerFrame = 0;
	void saveOriginalCopyOfMeshes();
};
#endif