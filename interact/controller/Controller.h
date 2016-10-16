#pragma once

// #include <QMainWindow>
// #include <QElapsedTimer>
// #include <QVector3D>
// #include <Leap.h>
//#include <qobject.h>
// typedef QVector<QVector<QVector3D>> TypeArray2;
// typedef QVector<QVector3D> TypeArray;
class OSVR_TimeValue;
class OSVR_OrientationReport;

#include <osvr/ClientKit/Context.h>
#include <osvr/ClientKit/Interface.h>

namespace osvr
{
	namespace clientkit{
		class ClientContext;
		//class Interface;
	}
}
class Controller{

	osvr::clientkit::ClientContext* context;

public:
	Controller();

	void Update();

	void GetLeftPosition(float &x, float &y, float &z);

	void GetLeftOrientation(float &w, float &x, float &y, float &z);

	void GetLeftButton(bool &pressed);

	void GetRightPosition(float &x, float &y, float &z);

	void GetRightOrientation(float &w, float &x, float &y, float &z);

	void GetRightButton(bool &pressed);

//private:
//	static void LeftOrientationCallback(void *userdata,
//		const OSVR_TimeValue *timestamp,
//		const OSVR_OrientationReport *report);
//
//	static void LeftPositionCallback(void *userdata,
//		const OSVR_TimeValue *timestamp,
//		const OSVR_PositionReport *report);

	// virtual void onFrame(const Leap::Controller & ctl);

 private:
	 osvr::clientkit::Interface leftButton;
	 osvr::clientkit::Interface rightButton;
	 osvr::clientkit::Interface leftHand;
	 osvr::clientkit::Interface rightHand;

	// QElapsedTimer *timer;

};