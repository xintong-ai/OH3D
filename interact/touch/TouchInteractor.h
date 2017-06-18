#ifndef TOUCHINTERACTOR_H
#define TOUCHINTERACTOR_H
#include <memory>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

#include <QtWidgets>
#include <GLWidget.h>


class TouchInteractor
{
public:
	TouchInteractor(){};
	~TouchInteractor(){};


	virtual bool TouchBeginEvent(QTouchEvent *event) { return false; };
	virtual bool TouchUpdateEvent(QTouchEvent *event) { return false; };
	virtual bool TouchEndEvent(QTouchEvent *event){ 
		actor->SetTouchInteractMode(TOUCH_NOT_START);
		return false;
	};

	virtual bool pinchTriggered(QPinchGesture *gesture) { return false; };

	void SetActor(GLWidget* _actor) {
		actor = _actor;
	}

protected:
	GLWidget* actor;
	int2 lastPt = make_int2(0, 0);

};
#endif