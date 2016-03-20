#ifndef LENS_RENDERABLE_H
#define LENS_RENDERABLE_H
#include "Renderable.h"


class Lens;
class SolidSphere;
class LensRenderable :public Renderable
{
	Q_OBJECT
	
	std::vector<Lens*> lenses;
	int pickedLens = -1;
	int2 lastPt = make_int2(0, 0);
	SolidSphere* lensCenterSphere;
public:
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void UpdateData() override;
	LensRenderable();
	std::vector<Lens*> GetLenses() { return lenses; }
	float3 GetBackLensCenter();
	void AddCircleLens();
	void AddLineLens();
	void AddLineBLens();
	void AddCurveBLens();

	void mousePress(int x, int y, int modifier) override;
	void mouseRelease(int x, int y, int modifier) override;
	void mouseMove(int x, int y, int modifier) override;
	bool MouseWheel(int x, int y, int modifier, int delta)  override;

	bool isSnapToGlyph = false;
	bool isSnapToFeature = false;

public slots:
	void SlotFocusSizeChanged(int v);
	void SlotSideSizeChanged(int v);
	void SlotDelLens();
	void SlotLensCenterChanged(float3 p);
	void adjustOffset();
	void RefineLensBoundary();
};
#endif