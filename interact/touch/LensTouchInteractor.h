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

public:
	LensTouchInteractor(){};
	~LensTouchInteractor(){};

	void SetLenses(std::vector<Lens*> *_lenses){ lenses = _lenses; }

	bool TouchBeginEvent(QTouchEvent *event) override;
	bool TouchUpdateEvent(QTouchEvent *event) override;
	bool pinchTriggered(QPinchGesture *gesture) override;

};
#endif