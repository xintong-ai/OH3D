#ifndef LENS_RENDERABLE_H
#define LENS_RENDERABLE_H
#include "Renderable.h"


class Lens;
class SolidSphere;
class LensRenderable :public Renderable
{
	Q_OBJECT

	std::vector<Lens*> *lenses;

	float3 lastLensCenter;
	float lastLensRatio = 0.3;
	bool lastLensCenterRecorded = false;

	int2 lastPt = make_int2(0, 0);
	SolidSphere* lensCenterSphere;

	bool highlightingCenter = false;
	bool highlightingMajorSide = false;
	bool highlightingMinorSide = false;
	bool highlightingCuboidFrame = false;

public:

	bool drawFullRetractor = false;	
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
	void DelLens();



	void SaveState(const char* filename);
	void LoadState(const char* filename);

public slots:
	//for keyboard
	void adjustOffset();
	void RefineLensBoundary();

};
#endif