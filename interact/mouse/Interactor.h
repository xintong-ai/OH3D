#ifndef INTERACTOR_H
#define INTERACTOR_H
#include <memory>
#include <vector_types.h>
#include <vector_functions.h>

class GLWidget;

class Interactor
{
public:
	bool isActive = true;

	Interactor(){};
	~Interactor(){};
	
	virtual void Rotate(float fromX, float fromY, float toX, float toY ){ return; };
	virtual void Translate(float x, float y){ return; };

	void SetActor(GLWidget* _actor) {
		actor = _actor;
	}

	virtual void mousePress(int x, int y, int modifier) {}
	virtual void mouseRelease(int x, int y, int modifier) {}
	virtual void mouseMove(int x, int y, int modifier) {}
	virtual bool MouseWheel(int x, int y, int modifier, float delta){ return false; };

protected:
	GLWidget* actor;
	int2 lastPt = make_int2(0, 0);


};
#endif