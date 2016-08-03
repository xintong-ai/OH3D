#include "DeformGLWidget.h"
#include <GLMatrixManager.h>
#include <LensRenderable.h>
#include <TransformFunc.h>

DeformGLWidget::DeformGLWidget(std::shared_ptr<GLMatrixManager> _matrixMgr,
	QWidget *parent) : GLWidget(_matrixMgr, parent)
{
	QTimer *aTimer = new QTimer;
	connect(aTimer, SIGNAL(timeout()), SLOT(animate()));
	aTimer->start(30);
}

void DeformGLWidget::animate()
{
	//for (auto renderer : renderers)
	//	renderer.second->animate();
	update();
}

void DeformGLWidget::UpdateDepthRange()
{
	float3 dataMin, dataMax;
	matrixMgr->GetVol(dataMin, dataMax);
	GLfloat modelview[16];
	GLfloat projection[16];
	matrixMgr->GetModelView(modelview);
	matrixMgr->GetProjection(projection, width, height);

	float4 p[8];
	p[0] = make_float4(dataMin.x, dataMin.y, dataMin.z, 1.0f);
	p[1] = make_float4(dataMin.x, dataMin.y, dataMax.z, 1.0f);
	p[2] = make_float4(dataMin.x, dataMax.y, dataMin.z, 1.0f);
	p[3] = make_float4(dataMin.x, dataMax.y, dataMax.z, 1.0f);
	p[4] = make_float4(dataMax.x, dataMin.y, dataMin.z, 1.0f);
	p[5] = make_float4(dataMax.x, dataMin.y, dataMax.z, 1.0f);
	p[6] = make_float4(dataMax.x, dataMax.y, dataMin.z, 1.0f);
	p[7] = make_float4(dataMax.x, dataMax.y, dataMax.z, 1.0f);

	float4 pClip[8];
	std::vector<float> clipDepths;
	for (int i = 0; i < 8; i++) {
		pClip[i] = Object2Clip(p[i], modelview, projection);
		clipDepths.push_back(pClip[i].z);
	}
	depthRange.x = clamp(*std::min_element(clipDepths.begin(), clipDepths.end()), 0.0f, 1.0f);
	depthRange.y = clamp(*std::max_element(clipDepths.begin(), clipDepths.end()), 0.0f, 1.0f);
	//std::cout << "depthRange: " << depthRange.x << "," << depthRange.y << std::endl;
}

void DeformGLWidget::mouseReleaseEvent(QMouseEvent *event)
{
	GLWidget::mouseReleaseEvent(event);

	UpdateDepthRange();
}

void DeformGLWidget::wheelEvent(QWheelEvent * event)
{
	GLWidget::wheelEvent(event);
	UpdateDepthRange();
}

bool DeformGLWidget::TouchBeginEvent(QTouchEvent *event)
{
	QList<QTouchEvent::TouchPoint> pts = event->touchPoints();
	QPointF p = pts.back().lastPos();
	//std::cout << "p:" << p.x() << "," << p.y() << std::endl;
	QPoint posGL = pixelPosToGLPos(QPoint(p.x(), p.y()));
	insideLens = ((LensRenderable*)renderers["lenses"])->InsideALens(posGL.x(), posGL.y());
	if (insideLens)
		SetInteractMode(INTERACT_MODE::MOVE_LENS);
	else
		SetInteractMode(INTERACT_MODE::TRANSFORMATION);
	//	SetInteractMode(INTERACT_MODE::NO_TRANSFORMATION);

	//std::cout << "pts.size(): " << pts.size() << std::endl;

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
			if (((LensRenderable*)renderers["lenses"])
				->OnLensInnerBoundary(make_int2(p1.x(), p1.y()), make_int2(p2.x(), p2.y()))){
				SetInteractMode(INTERACT_MODE::MODIFY_LENS_TWO_FINGERS);
			}
			else if (((LensRenderable*)renderers["lenses"])
				->TwoPointsInsideALens(make_int2(p1.x(), p1.y()), make_int2(p2.x(), p2.y()))){
				SetInteractMode(INTERACT_MODE::MODIFY_LENS_DEPTH);
			}
		}
		break;
	}
	case INTERACT_MODE::MODIFY_LENS_TWO_FINGERS:
	{
		if (2 == pts.size()) {
			QPointF p1 = pixelPosToGLPos(pts.at(0).lastPos());
			QPointF p2 = pixelPosToGLPos(pts.at(1).lastPos());
			((LensRenderable*)renderers["lenses"])
				->UpdateLensTwoFingers(make_int2(p1.x(), p1.y()), make_int2(p2.x(), p2.y()));
		}
		break;
	}
	}
	return true;
}

void DeformGLWidget::pinchTriggered(QPinchGesture *gesture)
{
	GLWidget::pinchTriggered(gesture);
	switch (GetInteractMode())
	{
		case INTERACT_MODE::MODIFY_LENS_DEPTH:
		{
			((LensRenderable*)renderers["lenses"])->ChangeLensDepth(gesture->totalScaleFactor() > 1 ? 26 : -26);
			break;
		}
	}
}

