#pragma once
#include <qobject.h>

class Controller;
class QController:public QObject
{
	Q_OBJECT
public:
	QController();

private:
	Controller *controller;

signals:
	void SignalUpdateControllers(QVector3D leftPos, QVector3D rightPos, 
		int numHands, bool pressed);
public slots:
	void Update();
};