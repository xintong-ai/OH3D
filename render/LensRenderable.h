#ifndef LENS_RENDERABLE_H
#define LENS_RENDERABLE_H
#include "Renderable.h"


class Lens;
class SolidSphere;
class LensRenderable :public Renderable
{
	Q_OBJECT


	float3 lastLensCenter;
	bool lastLensCenterRecorded = false;

	int pickedLens = -1;
	int2 lastPt = make_int2(0, 0);
	SolidSphere* lensCenterSphere;


	bool highlightingCenter = false;
	bool highlightingMajorSide = false;
	bool highlightingMinorSide = false;


	//used for Leap
	void ChangeLensCenterbyLeap(Lens *l, float3 p);
	void ChangeLensCenterbyTransferredLeap(Lens *l, float3 p);
	float3 GetTransferredLeapPos(float3 p);
	float3 prevPos, prevPos2, prevPointOfLens;
	float preForce;
	bool highlightingCuboidFrame = false;

public:
	int activedCursors = 0;
	int cursorColor[2];
	float3 cursorPos[2];


	bool drawFullRetractor = false;

	std::vector<Lens*> *lenses;

	float3 snapPos;
	
	bool changeLensWhenRotateData = true;
	bool drawInsicionOnCenterFace = false;

	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	LensRenderable(std::vector<Lens*>* _l);
	~LensRenderable();

	void AddCircleLens();
	void AddCircleLens3D();
	void AddLineLens();
	void AddLineLens3D();
	void AddCurveLens();

	void mousePress(int x, int y, int modifier) override;
	void mouseRelease(int x, int y, int modifier) override;
	void mouseMove(int x, int y, int modifier) override;
	bool MouseWheel(int x, int y, int modifier, int delta)  override;
	
	bool isSnapToGlyph = false;
	bool isSnapToFeature = false;

	//for touch screen
	void PinchScaleFactorChanged(float x, float y, float totalScaleFactor) override;
	void ChangeLensDepth(float v);
	bool InsideALens(int x, int y);
	bool TwoPointsInsideALens(int2 p1, int2 p2);
	bool OnLensInnerBoundary(int2 p1, int2 p2);
	void UpdateLensTwoFingers(int2 p1, int2 p2);

	void SaveState(const char* filename);
	void LoadState(const char* filename);

public slots:
	//for keyboard
	void SlotDelLens();
	void adjustOffset();
	void RefineLensBoundary();

//public slots: //those function are called by slot functions but are not slots themselves
public:
	void SlotOneHandChanged(float3 p);
	bool SlotOneHandChangedNew_lc(float3 thumpLeap, float3 indexLeap, float3 middleLeap, float3 ringLeap, float4 &markerPos, float &valRight, float &f);
	void SlotTwoHandChanged(float3 l, float3 r);
};
#endif