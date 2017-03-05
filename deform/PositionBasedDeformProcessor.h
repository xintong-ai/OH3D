#ifndef	POSITION_BASED_DEFORM_PROCESSOR_H
#define POSITION_BASED_DEFORM_PROCESSOR_H
#include <memory>
#include <vector>
#include <ctime>
#include <vector_types.h>

#include "Processor.h"

class Volume;
class MatrixManager;
class PositionBasedDeformProcessor :public Processor
{
public:
	std::shared_ptr<Volume> volume;
	std::shared_ptr<Volume> channelVolume;
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

	void doDeforme(float degree);
	void doTunnelDeforme(float degree);//generally degree is same with deformationScale
	void computeTunnelInfo();

	float3 tunnelStart, tunnelEnd;
	float deformationScale = 10; // for circle, it is maxRadius; for rect, the width of opening
	float deformationScale2nd = 7; // for rectangular, it is the other side length
	float3 rectDeformDir2nd; // for rectangular, it is the direction of deformationScale2nd

	float closeStartingRadius;

	bool hasBeenDeformed = false;
	bool hasOpenAnimeStarted = false;
	bool hasCloseAnimeStarted = false;
	std::clock_t startOpen;
	std::clock_t startClose;
	double totalDuration = 4;

};
#endif