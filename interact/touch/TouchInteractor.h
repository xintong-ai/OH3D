#ifndef TOUCHINTERACTOR_H
#define TOUCHINTERACTOR_H
#include <memory>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

class GLWidget;

class TouchInteractor
{
public:
	TouchInteractor(){};
	~TouchInteractor(){};


	void SetActor(GLWidget* _actor) {
		actor = _actor;
	}

protected:
	GLWidget* actor;

};
#endif