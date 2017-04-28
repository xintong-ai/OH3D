#ifndef LEAP_LISTENER_H
#define LEAP_LISTENER_H

#include <QMainWindow>
#include <QElapsedTimer>
#include <QVector3D>
#include <Leap.h>


#include "LeapInteractor.h"

typedef QVector<QVector<QVector3D>> TypeArray2;
typedef QVector<QVector3D> TypeArray;

class LeapInteractor;

class LeapListener : public QObject, public Leap::Listener {
	Q_OBJECT

signals:
	//void UpdateRectangle(QVector3D origin, QVector3D point1, QVector3D point2);
	void UpdateCamera(QVector3D origin, QVector3D xDir, QVector3D yDir, QVector3D zDir);
	//void UpdatePlane(QVector3D origin, QVector3D normal);
	void UpdateSkeletonHand(TypeArray2 fingers, TypeArray palm, float sphereRadius);
	void UpdateRightHand(QVector3D thumbTip, QVector3D indexTip, QVector3D indexDir);
	//void UpdateHands(QVector3D leftIndexTip, QVector3D rightIndexTip, int numHands);//original
	void UpdateHands(QVector3D rightThumbTip, QVector3D rightIndexTip, QVector3D leftThumbTip, QVector3D leftIndexTip, int numHands);
	void UpdateHandsNew(QVector3D rightThumbTip, QVector3D rightIndexTip, QVector3D leftThumbTip, QVector3D leftIndexTip, QVector3D rightMiddleTip, QVector3D rightRingTip, int numHands);
	void translate2(float v);
	void UpdateGesture(int gesture);

public:
	LeapListener()
	{
		//timer = new QTimer(this);
		timer = new QElapsedTimer();
		timer->start();
	};

	~LeapListener(){};

	virtual void onFrame(const Leap::Controller & ctl);

	void AddLeapInteractor(const char* name, void* r);

private:
	QElapsedTimer *timer;
protected:
	std::map<std::string, LeapInteractor*> interactors;
};

#endif	//LEAP_LISTENER_H