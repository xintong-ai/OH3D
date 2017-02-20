#ifndef	POSITION_BASED_DEFORM_PROCESSOR_H
#define POSITION_BASED_DEFORM_PROCESSOR_H
#include <memory>
#include <vector>
#include <ctime>

#include "Processor.h"

class Volume;
class MatrixManager;
class PositionBasedDeformProcessor :public Processor
{
public:
	std::shared_ptr<Volume> volume;
	std::shared_ptr<Volume> channelVolume;
//std::shared_ptr<MeshDeformProcessor> meshDeformer;
	std::shared_ptr<MatrixManager> matrixMgr;

	PositionBasedDeformProcessor(std::shared_ptr<Volume> ori, std::shared_ptr<MatrixManager> _m, std::shared_ptr<Volume> ch)
	{
		volume = ori;
		matrixMgr = _m;		
		channelVolume = ch;

		InitCudaSupplies();
	};

	~PositionBasedDeformProcessor(){
	};

	bool process(float* modelview, float* projection, int winWidth, int winHeight) override;

private:
	void InitCudaSupplies();

	bool hasDeformed = false;
	bool hasAnimeStarted = false;
	float maxRadius = 10;
	std::clock_t start;
	double totalDuration = 5;

	float3 tunnelStart, tunnelEnd;
	float closeStartingRadius;
};
#endif