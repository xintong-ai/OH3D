#ifndef LENSINTERACTOR_H
#define LENSINTERACTOR_H
#include <vector>
#include "Interactor.h"

class Lens;
class LensInteractor :public Interactor
{
private:

	float3 snapPos;

protected:
	std::vector<Lens*> *lenses = 0;
public:
	LensInteractor(){};
	~LensInteractor(){};

	void SetLenses(std::vector<Lens*> *_lenses){ lenses = _lenses; }

	void mousePress(int x, int y, int modifier) override;
	void mouseRelease(int x, int y, int modifier) override;
	void mouseMove(int x, int y, int modifier) override;
	bool MouseWheel(int x, int y, int modifier, int delta)  override;


	bool changeLensWhenRotateData = true; //view dependant or not
	bool isSnapToGlyph = false;
	bool isSnapToFeature = false;
};
#endif