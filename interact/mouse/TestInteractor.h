#ifndef tINTERACTOR_H
#define tINTERACTOR_H

#include "Interactor.h"

class TestInteractor :public Interactor
{
public:
	TestInteractor(){};
	~TestInteractor(){};

	void Rotate(float fromX, float fromY, float toX, float toY, std::shared_ptr<GLMatrixManager>) override ;
};
#endif