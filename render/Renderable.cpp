#include "Renderable.h"
#include "DataMgr.h"
#include "glwidget.h"
#include <helper_timer.h>

void Renderable::AllocOutImage() {
    if(h_output != NULL)
        delete [] h_output;
	int2 winSize = actor->GetWindowSize();
	h_output = new uint[winSize.x * winSize.y];
}

Renderable::~Renderable() {
	sdkDeleteTimer(&timer);
	if (h_output != NULL)
        delete [] h_output;
}

void Renderable::resize(int width, int height) {
    //winWidth = width;
    //winHeight = height;
    //AllocOutImage();
}

void Renderable::draw(float modelview[16], float projection[16])
{
}

void Renderable::DrawBegin()
{
#if ENABLE_TIMER
	sdkStartTimer(&timer);
#endif
}

void Renderable::DrawEnd(const char* rendererName)
{
#if ENABLE_TIMER
	sdkStopTimer(&timer);
	fpsCount++;
	if (fpsCount == fpsLimit)
	{
		//std::cout << rendererName << " time (ms):\t" << sdkGetAverageTimerValue(&timer) << std::endl;;
		fpsCount = 0;
		sdkResetTimer(&timer);
	}
#endif
}

Renderable::Renderable()
{
	sdkCreateTimer(&timer);
}


void Renderable::StartRenderableTimer()
{
	sdkStartTimer(&timer);
}


void Renderable::StopRenderableTimer()
{
	sdkStopTimer(&timer);
	fpsCount++;
	if (fpsCount == fpsLimit)
	{
		qDebug() << "Deform time (ms):\t" << sdkGetAverageTimerValue(&timer);
		fpsCount = 0;
		sdkResetTimer(&timer);
	}
}
