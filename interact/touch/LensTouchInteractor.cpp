#include "LensTouchInteractor.h"
#include "DeformGLWidget.h"
#include "glwidget.h"
#include "Lens.h"

bool LensTouchInteractor::TouchBeginEvent(QTouchEvent *event)
{
	if ((*lenses).size() < 1){
		//will support matrix operation later
		return false;
	}
	Lens* l = lenses->back();
	//currently only support circle lens in screen deform mode
	if (l->type != LENS_TYPE::TYPE_CIRCLE || ((DeformGLWidget*)actor)->GetDeformModel() != DEFORM_MODEL::SCREEN_SPACE){
		return false;
	}

	QList<QTouchEvent::TouchPoint> pts = event->touchPoints();

	if (1 == pts.size()){
		QPointF p = pts.back().lastPos();
		QPoint posGL = actor->pixelPosToGLPos(QPoint(p.x(), p.y()));

//		std::cout << "p:" << posGL.x() << "," << posGL.y() << std::endl;

		int x = posGL.x(), y = posGL.y();
		GLfloat modelview[16];
		GLfloat projection[16];
		actor->GetModelview(modelview);
		actor->GetProjection(projection);
		int2 winSize = actor->GetWindowSize();

		if (l->PointOnLensCenter(x, y, modelview, projection, winSize.x, winSize.y)) {
			actor->SetTouchInteractMode(TOUCH_INTERACT_MODE::TOUCH_MOVE_LENS);
		}
		else if (l->PointOnOuterBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
			actor->SetTouchInteractMode(TOUCH_INTERACT_MODE::TOUCH_MODIFY_LENS_TRANSITION_SIZE);
		}
		else if (l->PointOnInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
			actor->SetTouchInteractMode(TOUCH_INTERACT_MODE::TOUCH_MODIFY_LENS_FOCUS_SIZE);
		}

		lastPt = make_int2(x, y);
	}
	else if (2 == pts.size()){
		//this one seems not working. always get conducted to QGestures?

		QPointF p1 = actor->pixelPosToGLPos(pts.at(0).lastPos());
		QPointF p2 = actor->pixelPosToGLPos(pts.at(1).lastPos());
		std::cout << "p1:" << p1.x() << "," << p1.y() << std::endl;
		std::cout << "p2:" << p2.x() << "," << p2.y() << std::endl;
		std::cout << "pts.size():" << pts.size() << std::endl;

		return false;
	}

	return true;
}



bool LensTouchInteractor::TouchUpdateEvent(QTouchEvent *event)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	float3 posMin, posMax;
	actor->GetPosRange(posMin, posMax);
	
	//currently just for 1 finger case
	QList<QTouchEvent::TouchPoint> pts = event->touchPoints();
	QPointF p = pts.back().lastPos();
	QPoint posGL = actor->pixelPosToGLPos(QPoint(p.x(), p.y()));
	int x = posGL.x(), y = posGL.y();

	switch (actor->GetTouchInteractMode())
	{
	case TOUCH_INTERACT_MODE::TOUCH_MOVE_LENS:
	{
		float3 moveVec = lenses->back()->MoveLens(x, y, modelview, projection, winSize.x, winSize.y);
		if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE && lenses->back()->type == LENS_TYPE::TYPE_LINE){
			((LineLens3D*)lenses->back())->UpdateObjectLineLens(winSize.x, winSize.y, modelview, projection, posMin, posMax);
		}
				
		break;
	}
	case TOUCH_INTERACT_MODE::TOUCH_MODIFY_LENS_FOCUS_SIZE:
	{
		if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE)
			lenses->back()->ChangeLensSize(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);
		else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE){
			lenses->back()->ChangeObjectLensSize(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);

			if (lenses->back()->type == TYPE_LINE){
				((LineLens3D*)(lenses->back()))->UpdateObjectLineLens(winSize.x, winSize.y, modelview, projection, posMin, posMax);
			}
		}
		break;
	}
	case TOUCH_INTERACT_MODE::TOUCH_MODIFY_LENS_TRANSITION_SIZE:
	{
		if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE)
			lenses->back()->ChangeFocusRatio(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);
		else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE)
			lenses->back()->ChangeObjectFocusRatio(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);
		break;
		
	}
	}
	lastPt = make_int2(x, y);

	return true;
}




bool LensTouchInteractor::pinchTriggered(QPinchGesture *gesture)
{
	if ((*lenses).size() < 1){
		return false;
	}
	Lens* l = lenses->back();
	//currently only support circle lens in screen deform mode
	if (l->type != LENS_TYPE::TYPE_CIRCLE || ((DeformGLWidget*)actor)->GetDeformModel() != DEFORM_MODEL::SCREEN_SPACE ){
		return false;
	}


	QPoint posScreen = QPoint(gesture->centerPoint().x(), gesture->centerPoint().y()); //originally the position is regarding to the whole screen
	//std::cout << "posScreen:" << posScreen.x() << "," << posScreen.y() << std::endl;
	QPoint posWindowCorner = actor->window()->pos();
	//std::cout << "this->pos:" << actor->window()->pos().x() << "," << actor->window()->pos().y() << std::endl;
	QPoint posGL = actor->pixelPosToGLPos(posScreen - posWindowCorner);
	//std::cout << "posGL:" << posGL.x() << "," << posGL.y() << std::endl;
	int2 winSize = actor->GetWindowSize();
	if (posGL.x() < 0 || posGL.y() < 0 || posGL.x() > winSize.x - 1 || posGL.y() > winSize.y - 1){
		std::cout << "error getting the center position of the gesture!!!" << std::endl;
		return false;
	}

	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	

	TOUCH_INTERACT_MODE interactMode = actor->GetTouchInteractMode();
	int x = posGL.x(), y = posGL.y();
	if (interactMode == TOUCH_NOT_START)
	{
		if (l->PointInsideInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
			actor->SetTouchInteractMode(TOUCH_MODIFY_LENS_DEPTH);
		}
		else{
			if (gesture->changeFlags() & QPinchGesture::CenterPointChanged) {
				//COMING SOON
				//do something like matrix panning
			}
		}
	}
	else if (interactMode == TOUCH_MODIFY_LENS_DEPTH){
		float3 posMin, posMax;
		actor->GetPosRange(posMin, posMax);
		float3 dif = posMax - posMin;
		float coeff = min(min(dif.x, dif.y), dif.z) / 10.0 / 20.0 / 20.0;
		float delta = gesture->totalScaleFactor() > 1 ? 26 : -26;// gesture->totalScaleFactor() * 100;

		l->ChangeClipDepth(delta*coeff, modelview, projection);
	}
	
	if (gesture->state() == Qt::GestureFinished) {
		actor->SetTouchInteractMode(TOUCH_NOT_START);
	}
	
	return true;
}



