#ifndef LENS_RENDERABLE_H
#define LENS_RENDERABLE_H
#include "Renderable.h"


class Lens;
class LensRenderable :public Renderable
{
	Q_OBJECT
	
	std::vector<Lens*> lenses;
	int pickedLens = -1;
	//bool workingOnLens = false;
	int2 lastPt = make_int2(0, 0);
public:
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void UpdateData() override;
	LensRenderable(){}
	std::vector<Lens*> GetLenses() { return lenses; }
	//void AddSphereLens(int x, int y, int radius, float3 center);
	void AddCircleLens();
	void AddLineLens();

	//bool IsWorkingOnLens(){ return workingOnLens; }

	void mousePress(int x, int y, int modifier) override;
	void mouseRelease(int x, int y, int modifier) override;
	void mouseMove(int x, int y, int modifier) override;
	bool MouseWheel(int x, int y, int delta)  override;
public slots:
	void SlotFocusSizeChanged(int v);// { displace - (10 - v) * 0.1; }
	void SlotSideSizeChanged(int v);// { displace - (10 - v) * 0.1; }
};
#endif