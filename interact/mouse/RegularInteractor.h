

#ifndef REGULARINTERACTOR_H
#define REGULARINTERACTOR_H

#include "Interactor.h"

class RegularInteractor :public Interactor
{
public:
	RegularInteractor(){};
	~RegularInteractor(){};

	void Rotate(float fromX, float fromY, float toX, float toY, std::shared_ptr<GLMatrixManager>) override;

	void Translate(float x, float y, std::shared_ptr<GLMatrixManager>) override;
	void wheelEvent(float v, std::shared_ptr<GLMatrixManager>) override;


};
#endif