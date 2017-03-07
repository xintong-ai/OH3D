#ifndef tINTERACTOR_H
#define tINTERACTOR_H

#include <QVector3D>
#include <QMatrix4x4>

#include "MatrixInteractor.h"

class ImmersiveInteractor :public MatrixInteractor
{
	//not set
	QVector3D targetUpVecInLocal = QVector3D(0.0, 0.0, 1.0);
	void moveViewHorizontally(int d);
	void moveViewVertically(int d);

	void Rotate(float fromX, float fromY, float toX, float toY);
	void Translate(float x, float y);

public:
	ImmersiveInteractor(){};
	~ImmersiveInteractor(){};

	void mousePress(int x, int y, int modifier, int mouseKey = 0) override;
	bool MouseWheel(int x, int y, int modifier, float v) override;
	void mouseRelease(int x, int y, int modifier) override;
	void keyPress(char key) override;

	void mouseMoveMatrix(float fromX, float fromY, float toX, float toY, int modifier, int mouseKey) override;

	bool noMoveMode = false;
};
#endif