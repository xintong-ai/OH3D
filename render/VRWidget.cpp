#include "VRWidget.h"
#include "vector_types.h"
#include "vector_functions.h"
#include <iostream>
#include <helper_timer.h>
#include <glwidget.h>
#include <GlyphRenderable.h>


#include <osvr/ClientKit/ClientKit.h>
#include <osvr/ClientKit/Display.h>
#include <QMatrix4x4>
#include <GLMatrixManager.h>

#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()

VRWidget::VRWidget(std::shared_ptr<GLMatrixManager> _matrixMgr, QWidget *parent)
	: QOpenGLWidget(parent)
	, matrixMgr(_matrixMgr)
	, m_frame(0)
{
	//setFocusPolicy(Qt::StrongFocus);
	sdkCreateTimer(&timer);
}

void VRWidget::AddRenderable(const char* name, void* r)
{
	renderers[name] = (Renderable*)r;
	((Renderable*)r)->SetVRActor(this);
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
	initializeOpenGLFunctions();
	sdkCreateTimer(&timer);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);

	//Start OSVR and get OSVR display config

	ctx = std::make_shared<osvr::clientkit::ClientContext>("com.osvr.example.SDLOpenGL");
	display = std::make_shared<osvr::clientkit::DisplayConfig>(*ctx.get());
	if (!display->valid()) {
		std::cerr << "\nCould not get display config (server probably not "
			"running or not behaving), exiting."
			<< std::endl;
		return;
	}

	std::cout << "Waiting for the display to fully start up, including "
		"receiving initial pose update..."
		<< std::endl;
	while (!display->checkStartup()) {
		ctx->update();
	}
	std::cout << "OK, display startup status is good!" << std::endl;

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
	TimerStart();

	makeCurrent();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	ctx->update();

	QMatrix4x4 viewMat;
	display->getViewer(0).getEye(0).getViewMatrix(OSVR_MATRIX_COLMAJOR | OSVR_MATRIX_COLVECTORS,
		viewMat.data());
	matrixMgr->SetViewMat(viewMat);

	OSVR_ViewerCount viewers = display->getNumViewers();
	for (OSVR_ViewerCount viewer = 0; viewer < viewers; ++viewer) {
		OSVR_EyeCount eyes = display->getViewer(viewer).getNumEyes();
		for (OSVR_EyeCount eye = 0; eye < eyes; ++eye) {
			display->getViewer(viewer).getEye(eye).getViewMatrix(OSVR_MATRIX_COLMAJOR | OSVR_MATRIX_COLVECTORS,
				viewMat.data());
			QMatrix4x4 mv;// (modelview);
			//mv = mv.transposed();
			//mv = viewMat * mv;
			matrixMgr->GetModelViewMatrix(mv.data(), viewMat.data());
			OSVR_SurfaceCount surfaces = display->getViewer(viewer).getEye(eye).getNumSurfaces();
			for (OSVR_SurfaceCount surface = 0; surface < surfaces; ++surface) {
				auto viewport = display->getViewer(viewer).getEye(eye).getSurface(surface).getRelativeViewport();
				qgl->glViewport(static_cast<GLint>(viewport.left),
					static_cast<GLint>(viewport.bottom),
					static_cast<GLsizei>(viewport.width),
					static_cast<GLsizei>(viewport.height));

				/// Set the OpenGL projection matrix based on the one we
				/// computed.
				float zNear = 0.1;
				float zFar = 100;
				QMatrix4x4 projMat;
				display->getViewer(viewer).getEye(eye).getSurface(surface).getProjectionMatrix(
					zNear, zFar, OSVR_MATRIX_COLMAJOR | OSVR_MATRIX_COLVECTORS |
					OSVR_MATRIX_SIGNEDZ | OSVR_MATRIX_RHINPUT,
					projMat.data());

				//QMatrix4x4 pj(projection);
				//pj = pj.transposed();

				/// Call out to render our scene.
				for (auto renderer : renderers){
					//should use better methods than using renderer names...
					if (std::string(renderer.first).find("glyph") != std::string::npos || std::string(renderer.first).find("volume") != std::string::npos){
						renderer.second->drawVR(mv.data(), projMat.data(), eye);
					}
					else{
						renderer.second->draw(mv.data(), projMat.data());
					}
				}
			}
		}
	};

	TimerEnd();
}
//void Perspective(float fovyInDegrees, float aspectRatio,
//	float znear, float zfar);

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

void VRWidget::keyPressEvent(QKeyEvent * event)
{
	if (Qt::Key_F == event->key()){
		if (this->windowState() != Qt::WindowFullScreen){
			this->showFullScreen();
		}
		else{
			this->showNormal();
		}
	}
}

void VRWidget::UpdateGL()
{
	update();
}
