#include <controller/QController.h>
#include <controller/Controller.h>
#include <qvector3d.h>

QController::QController()
{
	controller = new Controller();
}

void QController::Update()
{
	controller->Update();
	QVector3D leftPos, rightPos;
	bool leftPressed, rightPressed, bothPressed;
	controller->GetLeftPosition(leftPos[0], leftPos[1], leftPos[2]);
	controller->GetRightPosition(rightPos[0], rightPos[1], rightPos[2]);
	controller->GetLeftButton(leftPressed);
	controller->GetRightButton(rightPressed);
	bothPressed = leftPressed && rightPressed;
	emit SignalUpdateControllers(leftPos, rightPos, 2, bothPressed);
}
