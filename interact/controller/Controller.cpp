//#include <qvector3d.h>
#include <iostream>

#include <osvr/ClientKit/Context.h>
#include <osvr/ClientKit/Interface.h>
#include <osvr/ClientKit/InterfaceStateC.h>

#include "Controller.h"

//void Controller::LeftOrientationCallback(void *userdata,
//                           const OSVR_TimeValue *timestamp,
//                           const OSVR_OrientationReport *report) {
//    // std::cout << "Got ORIENTATION report: Orientation = ("
//              // << osvrQuatGetW(&(report->rotation)) << ", ("
//              // << osvrQuatGetX(&(report->rotation)) << ", "
//              // << osvrQuatGetY(&(report->rotation)) << ", "
//              // << osvrQuatGetZ(&(report->rotation)) << "))" << std::endl;
//		emit UpdateLeftOrientation(
//				osvrQuatGetW(&(report->rotation)),
//				osvrQuatGetX(&(report->rotation)),
//				osvrQuatGetY(&(report->rotation)),
//				osvrQuatGetZ(&(report->rotation))
//			);
//}
//
//void Controller::LeftPositionCallback(void *userdata,
//                        const OSVR_TimeValue *timestamp,
//                        const OSVR_PositionReport *report) {
//    std::cout << "Got POSITION report: Position = (" << report->xyz.data[0]
//              << ", " << report->xyz.data[1] << ", " << report->xyz.data[2]
//              << ")" << std::endl;
//	emit UpdateLeftPosition(
//			report->xyz.data[0],
//			report->xyz.data[1],
//			report->xyz.data[2]
//		);
//}

Controller::Controller()
{
	context = new osvr::clientkit::ClientContext(
        "controller.listener");

	leftHand = context->getInterface("/me/hands/left");
	rightHand = context->getInterface("/me/hands/right");
	leftButton = context->getInterface("/com_osvr_Vive/Vive/button/6");
	rightButton = context->getInterface("/com_osvr_Vive/Vive/button/12");

    // This is just one of the paths. You can also use:
    // /me/hands/right
    // /me/head
    //osvr::clientkit::Interface lefthand =
    //    context->getInterface("/me/hands/left");
}

void Controller::GetLeftPosition(float &x, float &y, float &z)
{
	//osvr::clientkit::Interface hand =
	//	context->getInterface("/me/hands/left");
	OSVR_PoseState state;
	OSVR_TimeValue timestamp;
	OSVR_ReturnCode ret =
		osvrGetPoseState(leftHand.get(), &timestamp, &state);
	if (OSVR_RETURN_SUCCESS != ret) {
		std::cout << "No pose state!" << std::endl;
	}
	else {
		x = state.translation.data[0];
		y = state.translation.data[1];
		z = state.translation.data[2];
	}
}

void Controller::GetLeftOrientation(float &w, float &x, float &y, float &z)
{
	//osvr::clientkit::Interface hand =
	//	context->getInterface("/me/hands/left");
	OSVR_PoseState state;
	OSVR_TimeValue timestamp;
	OSVR_ReturnCode ret =
		osvrGetPoseState(leftHand.get(), &timestamp, &state);
	if (OSVR_RETURN_SUCCESS != ret) {
		std::cout << "No pose state!" << std::endl;
	}
	else {
		w = osvrQuatGetW(&(state.rotation));
		x = osvrQuatGetX(&(state.rotation));
		y = osvrQuatGetY(&(state.rotation));
		z = osvrQuatGetZ(&(state.rotation));
	}

}

void Controller::GetLeftButton(bool &pressed)
{
	//osvr::clientkit::Interface button =
	//	context->getInterface("/com_osvr_Vive/Vive/button/6");
	OSVR_ButtonState buttonState;
	OSVR_TimeValue timestamp;
	OSVR_ReturnCode ret2 =
		osvrGetButtonState(leftButton.get(), &timestamp, &buttonState);
	pressed = (OSVR_BUTTON_PRESSED == buttonState);
}

void Controller::GetRightPosition(float &x, float &y, float &z)
{

	OSVR_PoseState state;
	OSVR_TimeValue timestamp;
	OSVR_ReturnCode ret =
		osvrGetPoseState(rightHand.get(), &timestamp, &state);
	if (OSVR_RETURN_SUCCESS != ret) {
		std::cout << "No pose state!" << std::endl;
	}
	else {
		x = state.translation.data[0];
		y = state.translation.data[1];
		z = state.translation.data[2];
	}
}

void Controller::GetRightOrientation(float &w, float &x, float &y, float &z)
{
	//osvr::clientkit::Interface hand =
	//	context->getInterface("/me/hands/right");
	OSVR_PoseState state;
	OSVR_TimeValue timestamp;
	OSVR_ReturnCode ret =
		osvrGetPoseState(rightHand.get(), &timestamp, &state);
	if (OSVR_RETURN_SUCCESS != ret) {
		std::cout << "No pose state!" << std::endl;
	}
	else {
		w = osvrQuatGetW(&(state.rotation));
		x = osvrQuatGetX(&(state.rotation));
		y = osvrQuatGetY(&(state.rotation));
		z = osvrQuatGetZ(&(state.rotation));
	}

}

void Controller::GetRightButton(bool &pressed)
{

	OSVR_ButtonState buttonState;
	OSVR_TimeValue timestamp;
	OSVR_ReturnCode ret2 =
		osvrGetButtonState(rightButton.get(), &timestamp, &buttonState);
	pressed = (OSVR_BUTTON_PRESSED == buttonState);
}

void Controller::Update()
{
	context->update();
}

// void LeapListener::onFrame(const Leap::Controller & ctl)
// {
	// if(timer->elapsed() > 16.7)
	// {
			// emit UpdateHands(Leap2QVector(indexTipLeft), Leap2QVector(indexTipRight), 2);

		// timer->restart();
	// }
// }
