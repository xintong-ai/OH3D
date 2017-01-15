#ifndef LENSTOUCHINTERACTOR_H
#define LENSTOUCHINTERACTOR_H
#include <vector>
#include "TouchInteractor.h"

class Lens;
class Particle;
class LensTouchInteractor :public TouchInteractor
{
private:

	
	std::vector<Lens*> *lenses = 0;

	//for touch screen
	void PinchScaleFactorChanged(float x, float y, float totalScaleFactor);
	void ChangeLensDepth(float v);
	bool InsideALens(int x, int y);
	bool TwoPointsInsideALens(int2 p1, int2 p2);
	bool OnLensInnerBoundary(int2 p1, int2 p2);
	void UpdateLensTwoFingers(int2 p1, int2 p2);

public:
	LensTouchInteractor(){};
	~LensTouchInteractor(){};

	void SetLenses(std::vector<Lens*> *_lenses){ lenses = _lenses; }




};
#endif