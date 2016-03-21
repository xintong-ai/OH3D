#include "glwidget.h"
#include "vector_types.h"
#include "vector_functions.h"
#include "helper_math.h"
#include <iostream>
#include <fstream>
#include <helper_timer.h>
#include <Renderable.h>
#include <GlyphRenderable.h>
#include <VRWidget.h>
#include <TransformFunc.h>
#include <GLMatrixManager.h>
#include <LensRenderable.h>

GLWidget::GLWidget(std::shared_ptr<GLMatrixManager> _matrixMgr, QWidget *parent)
: QOpenGLWidget(parent)
    , m_frame(0)
	, matrixMgr(_matrixMgr)
{
    setFocusPolicy(Qt::StrongFocus);
    sdkCreateTimer(&timer);


	QTimer *aTimer = new QTimer;
	connect(aTimer, SIGNAL(timeout()), SLOT(animate()));
	aTimer->start(30);


	grabGesture(Qt::PinchGesture);

}


void GLWidget::AddRenderable(const char* name, void* r)
{
	renderers[name] = (Renderable*)r;
	((Renderable*)r)->SetAllRenderable(&renderers);
	((Renderable*)r)->SetActor(this);
	//((Renderable*)r)->SetWindowSize(width, height);
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
    sdkCreateTimer(&timer);
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
        //qDebug() << "FPS: "<<ifps;
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
    //glMatrixMode(GL_MODELVIEW);
    //glLoadIdentity();

    //glGetFloatv(GL_PROJECTION_MATRIX, projection);
	//glViewport(0, 0, (GLint)width, (GLint)height);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//((GlyphRenderable*)GetRenderable("glyph"))->SetDispalceOn(true);
	GLfloat modelview[16];
	GLfloat projection[16];
	matrixMgr->GetModelView(modelview);
	matrixMgr->GetProjection(projection, width, height);
	for (auto renderer : renderers)
	{
		renderer.second->draw(modelview, projection);
	}

    TimerEnd();
	if (nullptr != vrWidget){
		vrWidget->UpdateGL();
	}
}



void GLWidget::resizeGL(int w, int h)
{
    width = w;
    height = h;
	std::cout << "OpenGL window size:" << w << "x" << h << std::endl;
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
			matrixMgr->Rotate(from.x(), from.y(), to.x(), to.y());
		}
		else if (event->buttons() & Qt::RightButton) {
			QPointF diff = pixelPosToViewPos(pos) - pixelPosToViewPos(prevPos);
			matrixMgr->Translate(diff.x(), diff.y());
		}
	}
	QPoint posGL = pixelPosToGLPos(event->pos());
	for (auto renderer : renderers)
		renderer.second->mouseMove(posGL.x(), posGL.y(), QApplication::keyboardModifiers());

    prevPos = pos;
    update();
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
	std::cout << "mousePressEvent:" <<  std::endl;

	QPointF pos = event->pos();
	QPoint posGL = pixelPosToGLPos(event->pos());
	//lastPt = make_int2(posGL.x(), posGL.y());

	//if (pinching)
	//	return;
	makeCurrent();
	for (auto renderer : renderers)
		renderer.second->mousePress(posGL.x(), posGL.y(), QApplication::keyboardModifiers());

    prevPos = pos;
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
	//if (pinching)
	//	return;

	pinched = false;

	QPoint posGL = pixelPosToGLPos(event->pos());
	for (auto renderer : renderers)
		renderer.second->mouseRelease(posGL.x(), posGL.y(), QApplication::keyboardModifiers());

	UpdateDepthRange();
}

void GLWidget::wheelEvent(QWheelEvent * event)
{
	bool doTransform = true;
	QPoint posGL = pixelPosToGLPos(event->pos());
	for (auto renderer : renderers){
		if (renderer.second->MouseWheel(posGL.x(), posGL.y(), QApplication::keyboardModifiers(), event->delta()))
			doTransform = false;
	}
	if (doTransform){
		matrixMgr->Scale(event->delta());
		//transScale *= exp(event->delta() * -0.001);
		UpdateDepthRange();
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
		return TouchEvent(static_cast<QTouchEvent*>(event));
	return QWidget::event(event);
}

bool GLWidget::TouchEvent(QTouchEvent *event)
{
	QList<QTouchEvent::TouchPoint> pts = event->touchPoints();
	QPointF p = pts.back().lastPos();
	//std::cout << "p:" << p.x() << "," << p.y() << std::endl;
	QPoint posGL = pixelPosToGLPos(QPoint(p.x(), p.y()));
	insideLens = ((LensRenderable*)renderers["lenses"])->InsideALens(posGL.x(), posGL.y());

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

	if (insideLens){
		interactMode = INTERACT_MODE::MODIFY_LENS_DEPTH;
		for (auto renderer : renderers)
			renderer.second->PinchScaleFactorChanged(
			0,
			0,
			gesture->totalScaleFactor());
	}
	else {
		QPinchGesture::ChangeFlags changeFlags = gesture->changeFlags();
		//if (INTERACT_MODE::TRANSFORMATION == interactMode){
		if (changeFlags & QPinchGesture::ScaleFactorChanged) {
			//currentTransScale = gesture->totalScaleFactor();// exp(/*event->delta()*/gesture->totalScaleFactor() * 0.01);
			matrixMgr->SetCurrentScale(gesture->totalScaleFactor());
			update();
		}

		if (changeFlags & QPinchGesture::CenterPointChanged) {
			// transform only when there is no lens
			QPointF diff = pixelPosToViewPos(gesture->centerPoint())
				- pixelPosToViewPos(gesture->lastCenterPoint());
			//transVec[0] += diff.x();
			//transVec[1] += diff.y();
			matrixMgr->Translate(diff.x(), diff.y());
			update();
		}

		if (gesture->state() == Qt::GestureFinished) {
			matrixMgr->FinishedScale();
		}
		//}

	}
	if (gesture->state() == Qt::GestureFinished) {
		interactMode = INTERACT_MODE::TRANSFORMATION;
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

Renderable* GLWidget::GetRenderable(const char* name)
{
	if (renderers.find(name) == renderers.end()) {
		std::cout << "No renderer named : " << name << std::endl;
		exit(1);
	}
	return renderers[name];
}



void GLWidget::UpdateGL()
{
	update();
}

void GLWidget::animate()
{
	for (auto renderer : renderers)
		renderer.second->animate();
	update();
}


void GLWidget::UpdateDepthRange()
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

float3 GLWidget::DataCenter()
{
	return matrixMgr->DataCenter();
}

void GLWidget::GetModelview(float* m)
{
	matrixMgr->GetModelView(m);
}

void GLWidget::GetProjection(float* m)
{
	matrixMgr->GetProjection(m, width, height);
}

