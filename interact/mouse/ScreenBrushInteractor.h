#ifndef SCREENBRUSHINTERACTOR_H
#define SCREENBRUSHINTERACTOR_H
#include <vector>
#include "Interactor.h"

class ScreenMarker;
class ScreenBrushInteractor :public Interactor
{
private:
	bool isRemovingBrush = false;
	
	std::shared_ptr<ScreenMarker> sm;

public:
	ScreenBrushInteractor(){ isActive = false; };
	~ScreenBrushInteractor(){};

	void setScreenMarker(std::shared_ptr<ScreenMarker> _sm){ sm = _sm; }

	void mousePress(int x, int y, int modifier) override;
	void mouseRelease(int x, int y, int modifier) override;
	void mouseMove(int x, int y, int modifier) override;
	bool MouseWheel(int x, int y, int modifier, float delta)  override;

	
};
#endif