#include "glwidget.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <iostream>
#include <fstream>
#include <helper_timer.h>
#include <Windows.h>


#include "Renderable.h"
#include "VRWidget.h"
#include "GLMatrixManager.h"
#include "Processor.h"
#include "mouse/Interactor.h"

#ifdef USE_TOUCHSCREEN
#include "touch/TouchInteractor.h"
#endif

// removing the following lines will cause runtime error
#ifdef WIN32
#include <windows.h>
#endif
#define qgl	QOpenGLContext::currentContext()->functions()

GLWidget::GLWidget(std::shared_ptr<GLMatrixManager> _matrixMgr, QWidget *parent)
: QOpenGLWidget(parent), matrixMgr(_matrixMgr)
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
	//glClearColor(0.7f, 0.7f, 0.7f, 1.0f);
	//glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

	glEnable(GL_DEPTH_TEST);
}

void GLWidget::computeFPS()
{
    fpsCount++;
    if (fpsCount == fpsLimit)
    {
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        //qDebug() << "Processor plus Renderer FPS: "<<ifps;
        fpsCount = 0;
        fpsLimit = min(4*ifps, 100);
        sdkResetTimer(&timer);
    }
}

void GLWidget::TimerStart()
{
    sdkStartTimer(&timer);
}

void GLWidget::TimerEnd()
{
	fpsCount++;
	if (fpsCount >= fpsLimit)
	{
		sdkStopTimer(&timer);

		float ifps = 1.f*fpsCount / (sdkGetAverageTimerValue(&timer) / 1000.f);
		//qDebug() << "Overall FPS: "<<ifps;

		fpsCount = 0;
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
		fpsLimit = max(min(4 * ifps, 128), 16);
	}

	//computeFPS();
}


void GLWidget::paintGL() {
    /****transform the view direction*****/
	makeCurrent();
	//TimerStart();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	GLfloat modelview[16];
	GLfloat projection[16];
	matrixMgr->GetModelViewMatrix(modelview);
	matrixMgr->GetProjection(projection);


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

	//glEnable(GL_BLEND);

	for (auto renderer : renderers)
	{
		renderer.second->DrawBegin();
		renderer.second->draw(modelview, projection);
		renderer.second->DrawEnd(renderer.first.c_str());
	}

	//glDisable(GL_BLEND);

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

	if (needSaveImage){
		SYSTEMTIME st;
		GetSystemTime(&st);
		std::string fname = "screenShot_" + std::to_string(st.wMinute) + std::to_string(st.wSecond) + std::to_string(st.wMilliseconds) + ".png";
		std::cout << "saving screen shot to file: " << fname << std::endl;

		qgl->glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &screenTex);
		glBindTexture(GL_TEXTURE_2D, screenTex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);

		int w = width, h = height;

		std::cout << "w " << w << " h " << h << std::endl;
		uint8_t *pixels = new uint8_t[w * h * 3];
		// copy pixels from screen
		glBindTexture(GL_TEXTURE_2D, screenTex);
		glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, w, h);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid *)pixels);
	
		QImage image(w, h, QImage::Format_RGB32);
		for (int i = 0; i<w; ++i) {
			for (int j = 0; j<h; ++j) {
				int jinv = h - 1 - j;
				int ind = (i + j*w) * 3;
				image.setPixel(i, jinv, 256 * 256 * pixels[ind] + 256 * pixels[ind + 1] + pixels[ind + 2]);
			}
		}

		image.save(fname.c_str());

		glDeleteTextures(1, &screenTex);
		screenTex = 0;

		delete[]pixels;
		needSaveImage = false;
		std::cout << "finish saving image" << std::endl;
	}
}

void GLWidget::resizeGL(int w, int h)
{
    width = w;
    height = h;
	std::cout << "OpenGL window size:" << w << "x" << h << std::endl;
	
	matrixMgr->setWinSize(w, h);
	
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
}





void GLWidget::mousePressEvent(QMouseEvent *event)
{
	if (interactMode == UNDER_TOUCH)
		return;

	QPointF pos = event->pos();
	QPoint posGL = pixelPosToGLPos(event->pos());
	//lastPt = make_int2(posGL.x(), posGL.y());

	makeCurrent();

	int mouseKey = 0;
	if (event->buttons() & Qt::LeftButton){
		mouseKey = 1;
	}
	else if (event->buttons() & Qt::RightButton){
		mouseKey = 2;
	}
	for (auto interactor : interactors)
		interactor.second->mousePress(posGL.x(), posGL.y(), QApplication::keyboardModifiers(), mouseKey);

	prevPos = pos;
}

void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
	if (interactMode == UNDER_TOUCH)
		return;

	QPointF pos = event->pos();
	if (INTERACT_MODE::OPERATE_MATRIX == interactMode) {
		QPointF from = pixelPosToViewPos(prevPos);
		QPointF to = pixelPosToViewPos(pos);
		int mouseKey = 0;
		if (event->buttons() & Qt::LeftButton){
			mouseKey = 1;
		}
		else if (event->buttons() & Qt::RightButton){
			mouseKey = 2;
		}
		for (auto interactor : interactors)
			interactor.second->mouseMoveMatrix(from.x(), from.y(), to.x(), to.y(), QApplication::keyboardModifiers(), mouseKey);
	}
	else{
		QPoint posGL = pixelPosToGLPos(pos);
		for (auto interactor : interactors)
			interactor.second->mouseMove(posGL.x(), posGL.y(), QApplication::keyboardModifiers());
	}
	
	prevPos = pos;
    update();
}

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
	if (interactMode == UNDER_TOUCH || interactMode == OTHERS){
		interactMode = OPERATE_MATRIX;
		return;
	}

	QPoint posGL = pixelPosToGLPos(event->pos());
	for (auto interactor : interactors)
		interactor.second->mouseRelease(posGL.x(), posGL.y(), QApplication::keyboardModifiers());
}

void GLWidget::wheelEvent(QWheelEvent * event)
{
	QPoint posGL = pixelPosToGLPos(event->pos());

	//only the first interactor is executed, because it is tricky to apply INTERACT_MODE to monitor the operated object
	//so note the order when you add interactors

	for (auto interactor : interactors){
		if (interactor.second->MouseWheel(posGL.x(), posGL.y(), QApplication::keyboardModifiers(), event->delta()))
			break;
	}

	update();
}

void GLWidget::keyPressEvent(QKeyEvent * event)
{
	for (auto interactor : interactors)
		interactor.second->keyPress(event->key());
}


#ifdef USE_TOUCHSCREEN

void GLWidget::AddTouchInteractor(const char* name, void* r)
{
	touchInteractors[name] = (TouchInteractor*)r;
	((TouchInteractor*)r)->SetActor(this);
}

bool GLWidget::event(QEvent *event)
{
	if (event->type() == QEvent::Gesture){
		//for this branch, there is a small bug:
		//when two fingers are doing a gesture and then 1 finger leaves, the state will be changed to a touchUpdate or touchEnd,
		//without a proper touchBegin
		//currently not solving this problem yet
		//a save way would be abandon gesture, but detect the number of fingers use regular touch interactions

		interactMode = UNDER_TOUCH;

		QGestureEvent* eventG = static_cast<QGestureEvent*>(event);


		if (QGesture *swipe = eventG->gesture(Qt::SwipeGesture)){
			std::cout << "swipe \n" << std::endl;
			//			swipeTriggered(static_cast<QSwipeGesture *>(swipe));

		}
		else if (QGesture *pan = eventG->gesture(Qt::PanGesture)){
			std::cout << "pan \n" << std::endl;
			//			panTriggered(static_cast<QPanGesture *>(pan));

		}
		else if (QGesture *pinch = eventG->gesture(Qt::PinchGesture))
		{
			QPinchGesture* ges = static_cast<QPinchGesture *>(pinch);
			pinchTriggered(ges);

		}

		return true;	
	}
	else if (event->type() == QEvent::TouchBegin){
		interactMode = UNDER_TOUCH;
		return TouchBeginEvent(static_cast<QTouchEvent*>(event));
	}
	else if (event->type() == QEvent::TouchEnd){
		bool temp = TouchEndEvent(static_cast<QTouchEvent*>(event));
		interactMode = OTHERS;
		return temp;
	}
	else if (event->type() == QEvent::TouchUpdate)
		return TouchUpdateEvent(static_cast<QTouchEvent*>(event));

	return QWidget::event(event);
}

bool GLWidget::TouchBeginEvent(QTouchEvent *event)
{
	for (auto interactor : touchInteractors)
		interactor.second->TouchBeginEvent(event);
	return false;
}

bool GLWidget::TouchEndEvent(QTouchEvent *event)
{
	for (auto interactor : touchInteractors)
		interactor.second->TouchEndEvent(event);
	return true;
}

bool GLWidget::TouchUpdateEvent(QTouchEvent *event)
{
	for (auto interactor : touchInteractors)
		interactor.second->TouchUpdateEvent(event);
	return true;
}



void GLWidget::pinchTriggered(QPinchGesture *gesture)
{
	for (auto interactor : touchInteractors)
		interactor.second->pinchTriggered(gesture);

	return;
	/*
	if (gesture->state() == Qt::GestureFinished) {
		SetInteractMode(INTERACT_MODE::OPERATE_MATRIX);
	}
	*/
}
#endif



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
void GLWidget::GetDepthRange(float2 &dr)
{
	matrixMgr->GetClipDepthRangeOfVol(dr);
}

void GLWidget::GetModelview(float* m)
{
	matrixMgr->GetModelViewMatrix(m);
}

void GLWidget::GetProjection(float* m)
{
	matrixMgr->GetProjection(m);
}

void GLWidget::SetInteractMode(INTERACT_MODE v)
{ 
	interactMode = v; 
	//std::cout << "Set INTERACT_MODE: " << interactMode << std::endl; 
}

