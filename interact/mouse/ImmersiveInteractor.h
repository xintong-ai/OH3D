#ifndef tINTERACTOR_H
#define tINTERACTOR_H

#include "Interactor.h"

class ImmersiveInteractor :public Interactor
{
public:
	ImmersiveInteractor(){};
	~ImmersiveInteractor(){};

	void Rotate(float fromX, float fromY, float toX, float toY, std::shared_ptr<GLMatrixManager>) override ;

	void Translate(float x, float y, std::shared_ptr<GLMatrixManager>) override;
	void wheelEvent(float v, std::shared_ptr<GLMatrixManager>) override;
};
#endif