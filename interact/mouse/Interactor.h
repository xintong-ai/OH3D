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

	void SetActor(GLWidget* _actor) {
		actor = _actor;
	}

	virtual void mousePress(int x, int y, int modifier, int mouseKey = 0) {}
	virtual void mouseRelease(int x, int y, int modifier) {}
	virtual void mouseMove(int x, int y, int modifier) {}
	virtual void mouseMoveMatrix(float fromX, float fromY, float toX, float toY, int modifier, int mouseKey) {}
	virtual bool MouseWheel(int x, int y, int modifier, float delta){ return false; };
	virtual void keyPress(char key) {}

protected:
	GLWidget* actor;
	int2 lastPt = make_int2(0, 0);


};
#endif