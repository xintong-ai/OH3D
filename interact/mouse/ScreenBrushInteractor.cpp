#include "ScreenBrushInteractor.h"
#include "glwidget.h"
#include "TransformFunc.h"
#include "ScreenMarker.h"
#include <cuda_runtime.h>

const int cons=3;

void ScreenBrushInteractor::mousePress(int x, int y, int modifier, int mouseKey)
{
	if (!isActive)
		return;

	actor->SetInteractMode(INTERACT_MODE::OTHERS);
	int2 winSize = actor->GetWindowSize();

	for (int xx = x - cons; xx <= x + cons; xx++){
		for (int yy = y - cons; yy <= y + cons; yy++){
			if (xx >= 0 && xx < winSize.x && yy >= 0 && yy < winSize.y){
				if (isRemovingBrush){
					char c = 0;
					sm->isPixelSelected[yy*winSize.x + xx] = c;
					cudaMemcpy(&sm->dev_isPixelSelected[yy*winSize.x + xx], &c, sizeof(char)* 1, cudaMemcpyHostToDevice);
				}
				else{
					char c = 1;
					sm->isPixelSelected[yy*winSize.x + xx] = c;
					cudaMemcpy(&sm->dev_isPixelSelected[yy*winSize.x + xx], &c, sizeof(char)* 1, cudaMemcpyHostToDevice);
				}
			}
		}
	}
	sm->justChanged = true;
};

void ScreenBrushInteractor::mouseMove(int x, int y, int modifier)
{
	if (!isActive)
		return;

	int2 winSize = actor->GetWindowSize();

	for (int xx = x - cons; xx <= x + cons; xx++){
		for (int yy = y - cons; yy <= y + cons; yy++){
			if (xx >= 0 && xx < winSize.x && yy >= 0 && yy < winSize.y){
				if (isRemovingBrush){
					char c = 0;
					sm->isPixelSelected[yy*winSize.x + xx] = c;
					cudaMemcpy(&sm->dev_isPixelSelected[yy*winSize.x + xx], &c, sizeof(char)* 1, cudaMemcpyHostToDevice);
				}
				else{
					char c = 1;
					sm->isPixelSelected[yy*winSize.x + xx] = c;
					cudaMemcpy(&sm->dev_isPixelSelected[yy*winSize.x + xx], &c, sizeof(char)* 1, cudaMemcpyHostToDevice);
				}
			}
		}
	}
	sm->justChanged = true;

};

void ScreenBrushInteractor::mouseRelease(int x, int y, int modifier)
{
	if (!isActive)
		return;
	actor->SetInteractMode(OPERATE_MATRIX);

	sm->justChanged = true;
}

bool ScreenBrushInteractor::MouseWheel(int x, int y, int modifier, float delta)
{
	return false;
}