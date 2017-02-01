#include "LabelVolumeProcessor.h"
#include "ScreenMarker.h"

bool LabelVolumeProcessor::process(float modelview[16], float projection[16], int winW, int winH)
{
	if (!isActive)
		return false;
	
	if (sm->justChanged){
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