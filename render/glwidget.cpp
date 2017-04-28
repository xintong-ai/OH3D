#include "glwidget.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <iostream>
#include <fstream>
#include <helper_timer.h>

#include "Renderable.h"
#include "VRWidget.h"
#include "GLMatrixManager.h"
#include "Processor.h"
#include "mouse/Interactor.h"

GLWidget::GLWidget(std::shared_ptr<GLMatrixManager> _matrixMgr, QWidget *parent)
: QOpenGLWidget(parent)
    , m_frame(0)
	, matrixMgr(_matrixMgr)
{

    setFocusPolicy(Qt::StrongFocus);
    sdkCreateTimer(&timer);

	grabGesture(Qt::PinchGesture);

}


void GLWidget::AddRenderable(const char* name, void* r)
{
	renderers[name] = (Renderable*)r;
	((Renderable*)r)->SetActor(this);
}

void GLWidget::AddProcessor(const char* name, void* r)
{
	processors[name] = (Processor*)r;
	//((Processor*)r)->SetActor(this); //not sure if needed. better not rely on actor
}

void GLWidget::AddInteractor(const char* name, void* r)
{
	interactors[name] = (Interactor*)r;
	((Interactor*)r)->SetActor(this);
}

GLWidget::~GLWidget()
{
	sdkDeleteTimer(&timer);
}

QSize GLWidget::minimumSizeHint() const
{
	return QSize(256, 256);
}

QSize GLWidget::sizeHint() const
{
    return QSize(width, height);
}

void GLWidget::initializeGL()
{
	makeCurrent();
	initializeOpenGLFunctions();
	//glClearColor(0.2f, 0.2f, 0.4f, 1.0f);
	//glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	glEnable(GL_DEPTH_TEST);
}

void GLWidget::computeFPS()
{
    frameCount++;
    fpsCount++;
    if (fpsCount == fpsLimit)
    {
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        qDebug() << "FPS: "<<ifps;
        fpsCount = 0;
//        fpsLimit = (int)MAX(1.f, ifps);
        sdkResetTimer(&timer);
    }
}

void GLWidget::TimerStart()
{
    sdkStartTimer(&timer);
}

void GLWidget::TimerEnd()
{
	sdkStopTimer(&timer);
	computeFPS();
}


void GLWidget::paintGL() {
    /****transform the view direction*****/
	makeCurrent();
	TimerStart();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	GLfloat modelview[16];
	GLfloat projection[16];
	matrixMgr->GetModelViewMatrix(modelview);
	matrixMgr->GetProjection(projection, width, height);


	for (auto processor : processors)
	{
		processor.second->process(modelview, projection, width, height);
	}


#ifdef USE_LEAP
	if (blendOthers){ //only used for USE_LEAP
		for (auto renderer : renderers)
		{
			renderer.second->DrawBegin();

			if (std::string(renderer.first).find("Leap") != std::string::npos || std::string(renderer.first).find("lenses") != std::string::npos) //not a good design to use renderable names...
			{
				renderer.second->draw(modelview, projection);
				renderer.second->DrawEnd(renderer.first.c_str());
			}
			else
			{
				glEnable(GL_BLEND);
				glBlendColor(0.0f, 0.0f, 0.0f, 0.5f);
				glBlendFunc(GL_CONSTANT_ALPHA, GL_CONSTANT_ALPHA);

				renderer.second->draw(modelview, projection);
				renderer.second->DrawEnd(renderer.first.c_str());

				glDisable(GL_BLEND);
			}
		}
	}
	else{
		for (auto renderer : renderers)
		{
			renderer.second->DrawBegin();
			renderer.second->draw(modelview, projection);
			renderer.second->DrawEnd(renderer.first.c_str());
		}
	}
#else		
	for (auto renderer : renderers)
	{
		renderer.second->DrawBegin();
		renderer.second->draw(modelview, projection);
		renderer.second->DrawEnd(renderer.first.c_str());
	}
#endif


    TimerEnd();
#ifdef USE_OSVR
	if (nullptr != vrWidget){
		vrWidget->UpdateGL();
	}
#endif

#ifdef USE_CONTROLLER
	emit SignalPaintGL();
#endif
	
	UpdateGL();
}



void GLWidget::resizeGL(int w, int h)
{
    width = w;
    height = h;
	std::cout << "OpenGL window size:" << w << "x" << h << std::endl;

	for (auto processor : processors)
		processor.second->resize(w, h);

	for (auto renderer : renderers)
		renderer.second->resize(w, h);

    if(!initialized) {
        //make init here because the window size is not updated in InitiateGL()
		makeCurrent();
		for (auto renderer:renderers)
			renderer.second->init();
        initialized = true;
    }
    //glMatrixMode(GL_PROJECTION);
    //glLoadIdentity();
    //glMatrixMode(GL_MODELVIEW);
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
	if (pinching)
		return;

	QPointF pos = event->pos();
	if (INTERACT_MODE::TRANSFORMATION == interactMode) {
		if ((event->buttons() & Qt::LeftButton) && (!pinched)) {
			QPointF from = pixelPosToViewPos(prevPos);
			QPointF to = pixelPosToViewPos(pos);

			for (auto interactor : interactors)
				interactor.second->Rotate(from.x(), from.y(), to.x(), to.y());
			//matrixMgr->Rotate(from.x(), from.y(), to.x(), to.y());
		}
		else if (event->buttons() & Qt::RightButton) {
			QPointF diff = pixelPosToViewPos(pos) - pixelPosToViewPos(prevPos);

			for (auto interactor : interactors)
				interactor.second->Translate(diff.x(), diff.y());
			//matrixMgr->Translate(diff.x(), diff.y());
		}
	}
	QPoint posGL = pixelPosToGLPos(event->pos());
	//for (auto renderer : renderers)
		//renderer.second->mouseMove(posGL.x(), posGL.y(), QApplication::keyboardModifiers());
	for (auto interactor : interactors)
		interactor.second->mouseMove(posGL.x(), posGL.y(), QApplication::keyboardModifiers());

    prevPos = pos;
    update();
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
	QPointF pos = event->pos();
	QPoint posGL = pixelPosToGLPos(event->pos());
	//lastPt = make_int2(posGL.x(), posGL.y());

	//if (pinching)
	//	return;

	makeCurrent();
	//for (auto renderer : renderers)
		//	renderer.second->mousePress(posGL.x(), posGL.y(), QApplication::keyboardModifiers());

	for (auto interactor : interactors)
		interactor.second->mousePress(posGL.x(), posGL.y(), QApplication::keyboardModifiers());

    prevPos = pos;
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
	//if (pinching)
	//	return;

	pinched = false;

	QPoint posGL = pixelPosToGLPos(event->pos());
	//for (auto renderer : renderers)
	//	renderer.second->mouseRelease(posGL.x(), posGL.y(), QApplication::keyboardModifiers());
	for (auto interactor : interactors)
		interactor.second->mouseRelease(posGL.x(), posGL.y(), QApplication::keyboardModifiers());
}

void GLWidget::wheelEvent(QWheelEvent * event)
{
	bool doTransform = true;
	QPoint posGL = pixelPosToGLPos(event->pos());
	/*for (auto renderer : renderers){
		if (renderer.second->MouseWheel(posGL.x(), posGL.y(), QApplication::keyboardModifiers(), event->delta()))
			doTransform = false;
	}
	*/
	for (auto interactor : interactors){
		if (interactor.second->MouseWheel(posGL.x(), posGL.y(), QApplication::keyboardModifiers(), event->delta()))
			doTransform = false;
	}
	if (doTransform){
		for (auto interactor : interactors)
			interactor.second->wheelEvent(event->delta());
		//matrixMgr->Scale(event->delta());
	}
	update();
}

void GLWidget::keyPressEvent(QKeyEvent * event)
{
}

bool GLWidget::event(QEvent *event)
{
	if (event->type() == QEvent::Gesture)
		return gestureEvent(static_cast<QGestureEvent*>(event));
	else if (event->type() == QEvent::TouchBegin)
		return TouchBeginEvent(static_cast<QTouchEvent*>(event));
	else if (event->type() == QEvent::TouchEnd)
		return TouchEndEvent(static_cast<QTouchEvent*>(event));
	else if (event->type() == QEvent::TouchUpdate)
		return TouchUpdateEvent(static_cast<QTouchEvent*>(event));

	return QWidget::event(event);
}

//bool GLWidget::TouchBeginEvent(QTouchEvent *event)
//{
//	return false;
//}





bool GLWidget::TouchEndEvent(QTouchEvent *event)
{
	QList<QTouchEvent::TouchPoint> pts = event->touchPoints();
	//if (0 == pts.size()) {
	//SetInteractMode(INTERACT_MODE::TRANSFORMATION);
	//}
	return true;
}





//http://doc.qt.io/qt-5/gestures-overview.html
bool GLWidget::gestureEvent(QGestureEvent *event)
{
	if (QGesture *pinch = event->gesture(Qt::PinchGesture))
	{
		QPinchGesture* ges = static_cast<QPinchGesture *>(pinch);
		//QPointF center = ges->centerPoint());
		//std::cout << "center:" << center.x() << "," << center.y() << std::endl;
		pinchTriggered(ges);

	}
	return true;
}

void GLWidget::pinchTriggered(QPinchGesture *gesture/*, QPointF center*/)
{
	if (!pinching) {
		pinching = true;
		pinched = true;
	}
	QPoint gesScreen = QPoint(gesture->centerPoint().x(), gesture->centerPoint().y());

	//std::cout << "this->pos:" << this->pos().x() << "," << this->pos().y() << std::endl;
	//QPoint gesWin = gesScreen - this->pos();
	//QPoint posGL = pixelPosToGLPos(gesWin);
	////std::cout << "gesture->centerPoint():" << .x() << "," << gesture->centerPoint().y() << std::endl;
	//std::cout << "posGL:" << posGL.x() << "," << posGL.y() << std::endl;

	//if (insideLens){
	//	interactMode = INTERACT_MODE::MODIFY_LENS_DEPTH;
		//for (auto renderer : renderers)
		//	renderer.second->PinchScaleFactorChanged(
		//	0,
		//	0,
		//	gesture->totalScaleFactor());
	//}
	switch (interactMode)
	{

	case INTERACT_MODE::TRANSFORMATION:
	{
		//if (insideLens) break;
		QPinchGesture::ChangeFlags changeFlags = gesture->changeFlags();
		//if (INTERACT_MODE::TRANSFORMATION == interactMode){
		if (changeFlags & QPinchGesture::ScaleFactorChanged) {
			//currentTransScale = gesture->totalScaleFactor();// exp(/*event->delta()*/gesture->totalScaleFactor() * 0.01);
		//	matrixMgr->SetCurrentScale(gesture->totalScaleFactor());
			update();
		}

		if (changeFlags & QPinchGesture::CenterPointChanged) {
			// transform only when there is no lens
			QPointF diff = pixelPosToViewPos(gesture->centerPoint())
				- pixelPosToViewPos(gesture->lastCenterPoint());
			//transVec[0] += diff.x();
			//transVec[1] += diff.y();
			matrixMgr->TranslateInWorldSpace(diff.x(), diff.y());
			update();
		}

		if (gesture->state() == Qt::GestureFinished) {
			matrixMgr->FinishedScale();
		}
		break;
	}
	}

	if (gesture->state() == Qt::GestureFinished) {
		SetInteractMode(INTERACT_MODE::TRANSFORMATION);
		pinching = false;
	}
}

QPointF GLWidget::pixelPosToViewPos(const QPointF& p)
{
    return QPointF(2.0 * float(p.x()) / width - 1.0,
                   1.0 - 2.0 * float(p.y()) / height);
}

QPoint GLWidget::pixelPosToGLPos(const QPoint& p)
{
	return QPoint(p.x(), height - 1 - p.y());
}

QPoint GLWidget::pixelPosToGLPos(const QPointF& p)
{
	return QPoint(p.x(), height - 1 - p.y());
}



void GLWidget::UpdateGL()
{
	update();
}



float3 GLWidget::DataCenter()
{
	return matrixMgr->DataCenter();
}

void GLWidget::GetPosRange(float3 &pmin, float3 &pmax)
{
	matrixMgr->GetVol(pmin, pmax);
}

void GLWidget::GetModelview(float* m)
{
	matrixMgr->GetModelViewMatrix(m);
}

void GLWidget::GetProjection(float* m)
{
	matrixMgr->GetProjection(m, width, height);
}

void GLWidget::SetInteractMode(INTERACT_MODE v)
{ 
	interactMode = v; 
	//std::cout << "Set INTERACT_MODE: " << interactMode << std::endl; 
}
