#ifndef	POSITION_BASED_DEFORM_PROCESSOR_H
#define POSITION_BASED_DEFORM_PROCESSOR_H
#include <memory>
#include <vector>
#include <ctime>
#include <vector_types.h>

#include "Processor.h"

enum EYE_STATE { inCell, closeToWall, inWall };
enum VOLUME_STATE {DEFORMED, ORIGINAL};

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

	EYE_STATE lastEyeState = inCell;
	VOLUME_STATE lastVolumeState = ORIGINAL;

	//float3 transTunnelStart, transTunnelEnd;
	//float3 transRectVerticalDir; 
	// maybe equal to tunnelStart, tunnelEnd, rectVerticalDir?

	float deformationScale = 5;// 10; // for circle, it is maxRadius; for rect, the width of opening
	float deformationScaleVertical = 7; // for rectangular, it is the other side length

	void reset(){
		EYE_STATE lastEyeState = inCell;
		VOLUME_STATE lastVolumeState = ORIGINAL;
	}
private:

	bool inDeformedCell(float3 pos);

	std::clock_t startTime;

	float lastDeformationDegree;
	float3 lastDeformationDirVertical;
	float3 lastTunnelStart, lastTunnelEnd;

	float3 tunnelStart, tunnelEnd;
	float3 rectVerticalDir; // for rectangular, it is the direction of deformationScaleVertical

	void InitCudaSupplies();

	void doDeform(float degree);
	void doDeforme2Tunnel(float degree, float degreeClose);
	void doTunnelDeforme(float degree);
	void computeTunnelInfo(float3 centerPoint);


	bool hasBeenDeformed = false;
	bool hasOpenAnimeStarted = false;
	bool hasCloseAnimeStarted = false;
	std::clock_t startOpen;
	std::clock_t startClose;
	double totalDuration = 4;

	float closeStartingRadius;
	double closeDuration = 4;

	float3 targetUpVecInLocal = make_float3(0, 0, 1);	//note! the vector make_float3(0,0,1) may also be used in ImmersiveInteractor class

};
#endif