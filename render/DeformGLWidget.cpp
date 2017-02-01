#include "DeformGLWidget.h"
#include "GLMatrixManager.h"
#include "TransformFunc.h"

DeformGLWidget::DeformGLWidget(std::shared_ptr<GLMatrixManager> _matrixMgr,
	QWidget *parent) : GLWidget(_matrixMgr, parent)
{
	QTimer *aTimer = new QTimer;
	connect(aTimer, SIGNAL(timeout()), SLOT(animate()));
	aTimer->start(30);
}

void DeformGLWidget::animate()
{
	update(); 
}



bool DeformGLWidget::TouchBeginEvent(QTouchEvent *event)
{
	//QList<QTouchEvent::TouchPoint> pts = event->touchPoints();
	//QPointF p = pts.back().lastPos();
	////std::cout << "p:" << p.x() << "," << p.y() << std::endl;
	//QPoint posGL = pixelPosToGLPos(QPoint(p.x(), p.y()));
	//insideLens = ((LensRenderable*)renderers["lenses"])->InsideALens(posGL.x(), posGL.y());
	//if (insideLens)
	//	SetInteractMode(INTERACT_MODE::MOVE_LENS);
	//else
	//	SetInteractMode(INTERACT_MODE::TRANSFORMATION);
	////	SetInteractMode(INTERACT_MODE::NO_TRANSFORMATION);

	////std::cout << "pts.size(): " << pts.size() << std::endl;

	return true;
}

bool DeformGLWidget::TouchUpdateEvent(QTouchEvent *event)
{
	QList<QTouchEvent::TouchPoint> pts = event->touchPoints();
	switch(GetInteractMode())
	{
	case INTERACT_MODE::TRANSFORMATION:
	{
		if (2 == pts.size()) {
			QPointF p1 = pixelPosToGLPos(pts.at(0).lastPos());
			QPointF p2 = pixelPosToGLPos(pts.at(1).lastPos());
			//if (((LensRenderable*)renderers["lenses"])
			//	->OnLensInnerBoundary(make_int2(p1.x(), p1.y()), make_int2(p2.x(), p2.y()))){
			//	SetInteractMode(INTERACT_MODE::MODIFY_LENS_TWO_FINGERS);
			//}
			//else if (((LensRenderable*)renderers["lenses"])
			//	->TwoPointsInsideALens(make_int2(p1.x(), p1.y()), make_int2(p2.x(), p2.y()))){
			//	SetInteractMode(INTERACT_MODE::MODIFY_LENS_DEPTH);
			//}
		}
		break;
	}
	case INTERACT_MODE::MODIFY_LENS_TWO_FINGERS:
	{
		if (2 == pts.size()) {
			QPointF p1 = pixelPosToGLPos(pts.at(0).lastPos());
			QPointF p2 = pixelPosToGLPos(pts.at(1).lastPos());
			//((LensRenderable*)renderers["lenses"])
			//	->UpdateLensTwoFingers(make_int2(p1.x(), p1.y()), make_int2(p2.x(), p2.y()));
		}
		break;
	}
	}
	return true;
}

//!!! this function should not be placed here
void DeformGLWidget::pinchTriggered(QPinchGesture *gesture)
{
	GLWidget::pinchTriggered(gesture);
	switch (GetInteractMode())
	{
		case INTERACT_MODE::MODIFY_LENS_DEPTH:
		{
			//((LensRenderable*)renderers["lenses"])->ChangeLensDepth(gesture->totalScaleFactor() > 1 ? 26 : -26);
			break;
		}
	}
}

