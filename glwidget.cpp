#include "glwidget.h"
#include "vector_types.h"
#include "vector_functions.h"
#include "helper_math.h"
#include <iostream>
#include <fstream>
#include <helper_timer.h>
#include <Renderable.h>
#include <Trackball.h>
#include <Rotation.h>

GLWidget::GLWidget(QWidget *parent)
    : QOpenGLWidget(parent)
    , m_frame(0)
{
    setFocusPolicy(Qt::StrongFocus);
    sdkCreateTimer(&timer);

    trackball = new Trackball();
    rot = new Rotation();

    transRot.setToIdentity();

	grabGesture(Qt::PinchGesture);
}

void GLWidget::AddRenderable(const char* name, void* r)
{
	renderers[name] = (Renderable*)r;
	((Renderable*)r)->SetAllRenderable(&renderers);
	((Renderable*)r)->SetActor(this);
	((Renderable*)r)->SetWindowSize(width, height);
}

GLWidget::~GLWidget()
{
    cleanup();
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
    initializeOpenGLFunctions();
    sdkCreateTimer(&timer);
	//glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	//glClearColor(0.4f, 0.4f, 0.4f, 1.0f);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);

	//CHANGE_Huijie: initialize the FBO and the render buffer
	glGenRenderbuffers(2, renderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer[0]);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, width, height);

	glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer[1]);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);

	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);

	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
		GL_RENDERBUFFER, renderbuffer[0]);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
		GL_RENDERBUFFER, renderbuffer[1]);

}

void GLWidget::cleanup()
{
    sdkDeleteTimer(&timer);
	for (auto renderer:renderers)
		renderer.second->cleanup();
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
    TimerStart();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glTranslatef(transVec[0], transVec[1], transVec[2]);
    glMultMatrixf(transRot.data());
	glScalef(transScale * currentTransScale, transScale* currentTransScale, transScale* currentTransScale);

	float3 dataCenter = (dataMin + dataMax) * 0.5;
	float3 dataWidth = dataMax - dataMin;
    float dataMaxWidth = std::max(std::max(dataWidth.x, dataWidth.y), dataWidth.z);
	float scale = 2.0f / dataMaxWidth;
    glScalef(scale, scale, scale);
	glTranslatef(-dataCenter.x, -dataCenter.y, -dataCenter.z);

    GLfloat modelview[16];
    GLfloat projection[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
    glGetFloatv(GL_PROJECTION_MATRIX, projection);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	for (auto renderer : renderers)
		renderer.second->draw(modelview, projection);

	//CHANGE_Huijie: picking
	if (picking){
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		for (auto renderer : renderers)
			renderer.second->drawPicking(modelview, projection);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
		unsigned char cursorPixel[4];
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glReadPixels(xMouse, yMouse, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, cursorPixel);
		//std::cout << "cursor color: " << (int)cursorPixel[0] << " " << (int)cursorPixel[1] << " "
		//	<< (int)cursorPixel[2] << " " << (int)cursorPixel[3] << " " << std::endl;
		pickID = cursorPixel[0] + cursorPixel[1] * 256 + cursorPixel[2] * 256 * 256;
		std::cout << "pick id: " << pickID << std::endl;
		for (auto renderer : renderers)
			renderer.second->SetPickID(pickID);
		for (auto renderer : renderers)
			renderer.second->mousePress(xMouse, yMouse, QApplication::keyboardModifiers());
		picking = false;
		//glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	}

    TimerEnd();
}

void Perspective(float fovyInDegrees, float aspectRatio,
                      float znear, float zfar)
{
    float ymax, xmax;
    ymax = znear * tanf(fovyInDegrees * M_PI / 360.0);
    xmax = ymax * aspectRatio;
    glFrustum(-xmax, xmax, -ymax, ymax, znear, zfar);
}

void GLWidget::resizeGL(int w, int h)
{
    width = w;
    height = h;
	std::cout << "OpenGL window size:" << w << "x" << h << std::endl;
	for (auto renderer : renderers)
		renderer.second->resize(w, h);

	//CHANGE_Huijie
	glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer[0]);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, width, height);
	glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer[1]);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
		GL_RENDERBUFFER, renderbuffer[0]);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
		GL_RENDERBUFFER, renderbuffer[1]);

    if(!initialized) {
        //make init here because the window size is not updated in InitiateGL()
		//makeCurrent();
		for (auto renderer:renderers)
			renderer.second->init();
        initialized = true;
    }

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	Perspective(30, (float)width / height, (float)0.1, (float)10e4);
    glMatrixMode(GL_MODELVIEW);
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
	if (pinching)
		return;

	QPointF pos = event->pos();
	if ((event->buttons() & Qt::LeftButton) && (!pinched)) {
        QPointF from = pixelPosToViewPos(prevPos);
		QPointF to = pixelPosToViewPos(pos);
        *rot = trackball->rotate(from.x(), from.y(),
                                to.x(), to.y());
        float m[16];
        rot->matrix(m);
        QMatrix4x4 qm = QMatrix4x4(m).transposed();
        transRot = qm * transRot;

    } else if (event->buttons() & Qt::RightButton) {
        QPointF diff = pixelPosToViewPos(pos) - pixelPosToViewPos(prevPos);
        transVec[0] += diff.x();
        transVec[1] += diff.y();
    }
    prevPos = pos;
    update();
}

void GLWidget::mousePressEvent(QMouseEvent *event)
{
	if (pinching)
		return;

	QPointF pos = event->pos();
	QPoint posGL = pixelPosToGLPos(event->pos());

	//CHANGE_Huijie
	if (event->button() == Qt::LeftButton){
		xMouse = posGL.x();
		yMouse = posGL.y();
		picking = true;
	}

	/*for (auto renderer : renderers)
		renderer.second->mousePress(posGL.x(), posGL.y(), QApplication::keyboardModifiers());*/

    prevPos = pos;
	//CHANGE_Huijie
	update();
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
	if (pinching)
		return;

	pinched = false;

	QPoint posGL = pixelPosToGLPos(event->pos());
	for (auto renderer : renderers)
		renderer.second->mouseRelease(posGL.x(), posGL.y(), QApplication::keyboardModifiers());
}

void GLWidget::wheelEvent(QWheelEvent * event)
{
	transScale *= exp(event->delta() * -0.001);
	for (auto renderer : renderers)
		renderer.second->MouseWheel(event->delta());
	update();
}

void GLWidget::keyPressEvent(QKeyEvent * event)
{
}

bool GLWidget::event(QEvent *event)
{
	if (event->type() == QEvent::Gesture)
		return gestureEvent(static_cast<QGestureEvent*>(event));
	return QWidget::event(event);
}

//http://doc.qt.io/qt-5/gestures-overview.html
bool GLWidget::gestureEvent(QGestureEvent *event)
{
	if (QGesture *pinch = event->gesture(Qt::PinchGesture))
		pinchTriggered(static_cast<QPinchGesture *>(pinch));
	return true;
}

void GLWidget::pinchTriggered(QPinchGesture *gesture)
{
	if (!pinching) {
		pinching = true;
		pinched = true;
	}
	QPinchGesture::ChangeFlags changeFlags = gesture->changeFlags();
	if (changeFlags & QPinchGesture::ScaleFactorChanged) {
		currentTransScale = gesture->totalScaleFactor();// exp(/*event->delta()*/gesture->totalScaleFactor() * 0.01);
		update();
	}
	else {
		for (auto renderer : renderers)
			renderer.second->PinchScaleFactorChanged(gesture->totalScaleFactor());
	}
	if (changeFlags & QPinchGesture::CenterPointChanged) {
		// transform only when there is no lens
		QPointF diff = pixelPosToViewPos(gesture->centerPoint())
			- pixelPosToViewPos(gesture->lastCenterPoint());
		transVec[0] += diff.x();
		transVec[1] += diff.y();
		update();
	}

	if (gesture->state() == Qt::GestureFinished) {
		transScale *= currentTransScale;
		currentTransScale = 1;
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

void GLWidget::SetVol(int3 dim)
{ 
	dataMin = make_float3(0, 0, 0);
	dataMax = make_float3(dim.x - 1, dim.y - 1, dim.z - 1);
}

void GLWidget::SetVol(float3 posMin, float3 posMax)
{
	dataMin = posMin;
	dataMax = posMax;
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
