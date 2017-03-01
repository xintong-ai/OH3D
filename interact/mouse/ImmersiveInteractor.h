#ifndef tINTERACTOR_H
#define tINTERACTOR_H

#include <QVector3D>
#include <QMatrix4x4>

#include "MatrixInteractor.h"

class ImmersiveInteractor :public MatrixInteractor
{
	//not set
	QVector3D targetUpVecInLocal = QVector3D(0.0, 0.0, 1.0);
public:
	ImmersiveInteractor(){};
	~ImmersiveInteractor(){};

	void Rotate(float fromX, float fromY, float toX, float toY) override ;
	void Translate(float x, float y) override;
	bool MouseWheel(int x, int y, int modifier, float v) override;
};
#endif