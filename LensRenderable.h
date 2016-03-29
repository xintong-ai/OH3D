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

	void ChangeLensCenterbyLeap(float3 p);
public:
	float3 snapPos;
	void SnapLastLens();


	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void UpdateData() override;
	LensRenderable();
	std::vector<Lens*> GetLenses() { return lenses; }
	float3 GetBackLensCenter();
	float GetBackLensFocusRatio();
	float GetBackLensObjectRadius();
	void AddCircleLens();
	void AddLineLens();
	void AddLineBLens();
	void AddCurveBLens();

	void mousePress(int x, int y, int modifier) override;
	void mouseRelease(int x, int y, int modifier) override;
	void mouseMove(int x, int y, int modifier) override;
	bool MouseWheel(int x, int y, int modifier, int delta)  override;
	void PinchScaleFactorChanged(float x, float y, float totalScaleFactor) override;
	bool InsideALens(int x, int y);

	bool isSnapToGlyph = false;
	bool isSnapToFeature = false;

public slots:
	void SlotFocusSizeChanged(int v);
	void SlotSideSizeChanged(int v);
	void SlotDelLens();
	void SlotOneHandChanged(float3 p);
	void SlotTwoHandChanged(float3 l, float3 r);
	void adjustOffset();
	void RefineLensBoundary();
};
#endif