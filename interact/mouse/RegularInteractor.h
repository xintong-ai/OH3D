#ifndef REGULARINTERACTOR_H
#define REGULARINTERACTOR_H

#include "MatrixInteractor.h"

class RegularInteractor :public MatrixInteractor
{

	void Rotate(float fromX, float fromY, float toX, float toY);
	void Translate(float x, float y);

public:
	RegularInteractor(){};
	~RegularInteractor(){};

	bool MouseWheel(int x, int y, int modifier, float v) override;
	void mouseMoveMatrix(float fromX, float fromY, float toX, float toY, int modifier, int mouseKey) override;
};
#endif