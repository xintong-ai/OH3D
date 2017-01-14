#ifndef tINTERACTOR_H
#define tINTERACTOR_H

#include "MatrixInteractor.h"

class ImmersiveInteractor :public MatrixInteractor
{
public:
	ImmersiveInteractor(){};
	~ImmersiveInteractor(){};

	void Rotate(float fromX, float fromY, float toX, float toY) override ;

	void Translate(float x, float y) override;
	void wheelEvent(float v) override;
};
#endif