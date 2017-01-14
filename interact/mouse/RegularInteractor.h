

#ifndef REGULARINTERACTOR_H
#define REGULARINTERACTOR_H

#include "MatrixInteractor.h"

class RegularInteractor :public MatrixInteractor
{
public:
	RegularInteractor(){};
	~RegularInteractor(){};

	void Rotate(float fromX, float fromY, float toX, float toY) override;

	void Translate(float x, float y) override;
	void wheelEvent(float v) override;


};
#endif