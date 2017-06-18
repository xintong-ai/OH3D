#include "LabelVolumeProcessor.h"
#include "ScreenMarker.h"
#include "VolumeRenderableCUDAKernel.h"
#include "transformFunc.h"

bool LabelVolumeProcessor::process(float modelview[16], float projection[16], int winW, int winH)
{
	if (!isActive)
		return false;
	
	if (sm->justChanged){

		float _invmv[16];
		invertMatrix(modelview, _invmv);
		float3 eyeInLocal = make_float3(Camera2Object(make_float4(0, 0, 0, 1), _invmv));


		LabelProcessor(winW, winH, rcp->density, rcp->brightness, eyeInLocal, make_int3(labelVolume->size.width, labelVolume->size.height, labelVolume->size.depth), rcp->maxSteps, rcp->tstep, rcp->useColor, sm->dev_isPixelSelected, labelVolume.get());
		
		sm->justChanged = false;

		
		return true;
	}
	else{
		return true;
	}
}

void LabelVolumeProcessor::resize(int width, int height)
{
	sm->initMaskPixel(width, height);
}