#ifndef LEAP_LISTENER_H
#define LEAP_LISTENER_H

#include <QMainWindow>
#include <QElapsedTimer>
#include <QVector3D>
#include <Leap.h>

typedef QVector<QVector<QVector3D>> TypeArray2;
typedef QVector<QVector3D> TypeArray;

class LeapListener : public QObject, public Leap::Listener {
	Q_OBJECT

signals:
	//void UpdateRectangle(QVector3D origin, QVector3D point1, QVector3D point2);
	void UpdateCamera(QVector3D origin, QVector3D xDir, QVector3D yDir, QVector3D zDir);
	//void UpdatePlane(QVector3D origin, QVector3D normal);
	void UpdateSkeletonHand(TypeArray2 fingers, TypeArray palm, float sphereRadius);
	void UpdateRightHand(QVector3D thumbTip, QVector3D indexTip, QVector3D indexDir);
	void translate2(float v);
	void UpdateGesture(int gesture);

public:
	LeapListener()
	{
		//timer = new QTimer(this);
		timer = new QElapsedTimer();
		timer->start();
	}

	virtual void onFrame(const Leap::Controller & ctl);

private:
	QElapsedTimer *timer;


};

#endif	//LEAP_LISTENER_H