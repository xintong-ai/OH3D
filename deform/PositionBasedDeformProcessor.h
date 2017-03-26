#ifndef	POSITION_BASED_DEFORM_PROCESSOR_H
#define POSITION_BASED_DEFORM_PROCESSOR_H
#include <memory>
#include <vector>
#include <ctime>
#include <vector_types.h>
#include <helper_timer.h>

#include "Processor.h"
#include "Volume.h"

enum EYE_STATE { inCell, inWall };
enum VOLUME_STATE { ORIGINAL, DEFORMED};


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
		spacing = volume->spacing;
		InitCudaSupplies();		
		sdkCreateTimer(&timer);
		sdkCreateTimer(&timerFrame);
	};

	~PositionBasedDeformProcessor(){
		sdkDeleteTimer(&timer);
		sdkDeleteTimer(&timerFrame);
	};		


	bool process(float* modelview, float* projection, int winWidth, int winHeight) override;

	EYE_STATE lastEyeState = inCell;
	VOLUME_STATE lastVolumeState = ORIGINAL;

	float deformationScale = 5; // for rect, the width of opening
	float deformationScaleVertical = 7; // for rectangular, it is the other side length

	void reset(){
		EYE_STATE lastEyeState = inCell;
		VOLUME_STATE lastVolumeState = ORIGINAL;
	}

	bool hasOpenAnimeStarted = false;
	bool hasCloseAnimeStarted = false;

	float3 tunnelStart, tunnelEnd;
	float3 rectVerticalDir; // for rectangular, it is the direction of deformationScaleVertical

private:
	VolumeCUDA volumeCudaIntermediate; //when mixing opening and closing, an intermediate volume is needed

	float3 spacing;
	bool inDeformedCell(float3 pos);

	float lastOpenFinalDegree;
	float3 lastDeformationDirVertical;
	float3 lastTunnelStart, lastTunnelEnd;


	void InitCudaSupplies();

	void doDeform(float degree);
	void doDeform2Tunnel(float degree, float degreeClose);
	void doTunnelDeform(float degree);
	void computeTunnelInfo(float3 centerPoint);

	bool hasBeenDeformed = false;

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