#include "VRwidget.h"
#include "vector_types.h"
#include "vector_functions.h"
#include "helper_math.h"
#include <iostream>
#include <fstream>
#include <helper_timer.h>
#include <Renderable.h>
#include <Trackball.h>
#include <Rotation.h>
#include <glwidget.h>
#include <GlyphRenderable.h>

VRWidget::VRWidget(GLWidget* _mainGLWidget, QWidget *parent)
	: QOpenGLWidget(parent)
	, m_frame(0)
	, mainGLWidget(_mainGLWidget)
{
	setFocusPolicy(Qt::StrongFocus);
	sdkCreateTimer(&timer);

	//trackball = new Trackball();
	//rot = new Rotation();

	//QTimer *aTimer = new QTimer;
	//connect(aTimer, SIGNAL(timeout()), SLOT(animate()));
	//aTimer->start(30);

	//transRot.setToIdentity();

	//grabGesture(Qt::PinchGesture);
}

void VRWidget::AddRenderable(const char* name, void* r)
{
	renderers[name] = (Renderable*)r;
	((Renderable*)r)->SetAllRenderable(&renderers);
	//((Renderable*)r)->SetActor(this);
	//((Renderable*)r)->SetWindowSize(width, height);
}

VRWidget::~VRWidget()
{
	sdkDeleteTimer(&timer);
}

QSize VRWidget::minimumSizeHint() const
{
	return QSize(256, 256);
}

QSize VRWidget::sizeHint() const
{
	return QSize(width, height);
}

void VRWidget::initializeGL()
{
	//QOpenGLContext* a = mainGLWidget->context();
	//QOpenGLContext* b = this->context();
	//this->context()->setShareContext(mainGLWidget->context());
	initializeOpenGLFunctions();
	sdkCreateTimer(&timer);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);

}

void VRWidget::computeFPS()
{
	frameCount++;
	fpsCount++;
	if (fpsCount == fpsLimit)
	{
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		qDebug() << "FPS: " << ifps;
		fpsCount = 0;
		//        fpsLimit = (int)MAX(1.f, ifps);
		sdkResetTimer(&timer);
	}
}

void VRWidget::TimerStart()
{
	sdkStartTimer(&timer);
}

void VRWidget::TimerEnd()
{
	sdkStopTimer(&timer);
	computeFPS();
}


void VRWidget::paintGL() {
	/****transform the view direction*****/
	//wglMakeCurrent();
	TimerStart();
	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();

	//glTranslatef(transVec[0], transVec[1], transVec[2]);
	//glMultMatrixf(transRot.data());
	//glScalef(transScale * currentTransScale, transScale* currentTransScale, transScale* currentTransScale);

	//float3 dataCenter = (dataMin + dataMax) * 0.5;
	//float3 dataWidth = dataMax - dataMin;
	//float dataMaxWidth = std::max(std::max(dataWidth.x, dataWidth.y), dataWidth.z);
	//float scale = 2.0f / dataMaxWidth;
	//glScalef(scale, scale, scale);
	//glTranslatef(-dataCenter.x, -dataCenter.y, -dataCenter.z);

	GLfloat modelview[16];
	GLfloat projection[16];
	//glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
	//glGetFloatv(GL_PROJECTION_MATRIX, projection);
	mainGLWidget->GetModelview(modelview);
	mainGLWidget->GetProjection(projection);
	//glViewport(0, 0, (GLint)width, (GLint)height);
	makeCurrent();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//((GlyphRenderable*)GetRenderable("glyph"))->SetDispalceOn(false);
	for (auto renderer : renderers)
		renderer.second->draw(modelview, projection);

	TimerEnd();
}
//void Perspective(float fovyInDegrees, float aspectRatio,
//	float znear, float zfar);
//
//void Perspective(float fovyInDegrees, float aspectRatio,
//	float znear, float zfar)
//{
//	float ymax, xmax;
//	ymax = znear * tanf(fovyInDegrees * M_PI / 360.0);
//	xmax = ymax * aspectRatio;
//	glFrustum(-xmax, xmax, -ymax, ymax, znear, zfar);
//}

void VRWidget::resizeGL(int w, int h)
{
	width = w;
	height = h;
	std::cout << "OpenGL window size:" << w << "x" << h << std::endl;
	for (auto renderer : renderers)
		renderer.second->resize(w, h);

	if (!initialized) {
		//make init here because the window size is not updated in InitiateGL()
		//makeCurrent();
		makeCurrent();
		for (auto renderer : renderers)
			renderer.second->init();
		initialized = true;
	}

	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity();
	//Perspective(30, (float)width / height, (float)0.1, (float)10e4);
	//glMatrixMode(GL_MODELVIEW);
}

//void GLWidget::mouseMoveEvent(QMouseEvent *event)
//{
//	if (pinching)
//		return;
//
//	QPointF pos = event->pos();
//	if (INTERACT_MODE::TRANSFORMATION == interactMode) {
//		if ((event->buttons() & Qt::LeftButton) && (!pinched)) {
//			QPointF from = pixelPosToViewPos(prevPos);
//			QPointF to = pixelPosToViewPos(pos);
//			*rot = trackball->rotate(from.x(), from.y(),
//				to.x(), to.y());
//			float m[16];
//			rot->matrix(m);
//			QMatrix4x4 qm = QMatrix4x4(m).transposed();
//			transRot = qm * transRot;
//		}
//		else if (event->buttons() & Qt::RightButton) {
//			QPointF diff = pixelPosToViewPos(pos) - pixelPosToViewPos(prevPos);
//			transVec[0] += diff.x();
//			transVec[1] += diff.y();
//		}
//	}
//	QPoint posGL = pixelPosToGLPos(event->pos());
//	for (auto renderer : renderers)
//		renderer.second->mouseMove(posGL.x(), posGL.y(), QApplication::keyboardModifiers());
//
//	prevPos = pos;
//	update();
//}
//
//void GLWidget::mousePressEvent(QMouseEvent *event)
//{
//	if (pinching)
//		return;
//
//	QPointF pos = event->pos();
//	QPoint posGL = pixelPosToGLPos(event->pos());
//
//	for (auto renderer : renderers)
//		renderer.second->mousePress(posGL.x(), posGL.y(), QApplication::keyboardModifiers());
//
//	prevPos = pos;
//}
//
//void GLWidget::mouseReleaseEvent(QMouseEvent *event)
//{
//	if (pinching)
//		return;
//
//	pinched = false;
//
//	QPoint posGL = pixelPosToGLPos(event->pos());
//	for (auto renderer : renderers)
//		renderer.second->mouseRelease(posGL.x(), posGL.y(), QApplication::keyboardModifiers());
//}
//
//void GLWidget::wheelEvent(QWheelEvent * event)
//{
//	bool doTransform = true;
//	QPoint posGL = pixelPosToGLPos(event->pos());
//	for (auto renderer : renderers){
//		if (renderer.second->MouseWheel(posGL.x(), posGL.y(), event->delta()))
//			doTransform = false;
//	}
//	if (doTransform){
//		transScale *= exp(event->delta() * -0.001);
//	}
//	update();
//}

void VRWidget::keyPressEvent(QKeyEvent * event)
{
}

//bool VRWidget::event(QEvent *event)
//{
//	if (event->type() == QEvent::Gesture)
//		return gestureEvent(static_cast<QGestureEvent*>(event));
//	return QWidget::event(event);
//}

////http://doc.qt.io/qt-5/gestures-overview.html
//bool GLWidget::gestureEvent(QGestureEvent *event)
//{
//	if (QGesture *pinch = event->gesture(Qt::PinchGesture))
//		pinchTriggered(static_cast<QPinchGesture *>(pinch));
//	return true;
//}
//
//void GLWidget::pinchTriggered(QPinchGesture *gesture)
//{
//	if (!pinching) {
//		pinching = true;
//		pinched = true;
//	}
//	QPinchGesture::ChangeFlags changeFlags = gesture->changeFlags();
//	if (changeFlags & QPinchGesture::ScaleFactorChanged) {
//		currentTransScale = gesture->totalScaleFactor();// exp(/*event->delta()*/gesture->totalScaleFactor() * 0.01);
//		update();
//	}
//	else {
//		for (auto renderer : renderers)
//			renderer.second->PinchScaleFactorChanged(gesture->totalScaleFactor());
//	}
//	if (changeFlags & QPinchGesture::CenterPointChanged) {
//		// transform only when there is no lens
//		QPointF diff = pixelPosToViewPos(gesture->centerPoint())
//			- pixelPosToViewPos(gesture->lastCenterPoint());
//		transVec[0] += diff.x();
//		transVec[1] += diff.y();
//		update();
//	}
//
//	if (gesture->state() == Qt::GestureFinished) {
//		transScale *= currentTransScale;
//		currentTransScale = 1;
//		pinching = false;
//	}
//}
//
//QPointF GLWidget::pixelPosToViewPos(const QPointF& p)
//{
//	return QPointF(2.0 * float(p.x()) / width - 1.0,
//		1.0 - 2.0 * float(p.y()) / height);
//}
//
//QPoint GLWidget::pixelPosToGLPos(const QPoint& p)
//{
//	return QPoint(p.x(), height - 1 - p.y());
//}

Renderable* VRWidget::GetRenderable(const char* name)
{
	if (renderers.find(name) == renderers.end()) {
		std::cout << "No renderer named : " << name << std::endl;
		exit(1);
	}
	return renderers[name];
}

//void GLWidget::SetVol(int3 dim)
//{
//	dataMin = make_float3(0, 0, 0);
//	dataMax = make_float3(dim.x - 1, dim.y - 1, dim.z - 1);
//}
//
//void GLWidget::SetVol(float3 posMin, float3 posMax)
//{
//	dataMin = posMin;
//	dataMax = posMax;
//}
//


void VRWidget::UpdateGL()
{
	update();
}

//void VRWidget::animate()
//{
//	for (auto renderer : renderers)
//		renderer.second->animate();
//	update();
//}

//float3 VRWidget::DataCenter()
//{
//	return (dataMin + dataMax) * 0.5;
//}
