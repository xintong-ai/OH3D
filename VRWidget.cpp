#include "VRwidget.h"
#include "vector_types.h"
#include "vector_functions.h"
#include <iostream>
#include <helper_timer.h>
#include <glwidget.h>
#include <GlyphRenderable.h>


#include <osvr/ClientKit/ClientKit.h>
#include <osvr/ClientKit/Display.h>

VRWidget::VRWidget(GLWidget* _mainGLWidget, QWidget *parent)
	: QOpenGLWidget(parent)
	, m_frame(0)
	, mainGLWidget(_mainGLWidget)
{
	//setFocusPolicy(Qt::StrongFocus);
	sdkCreateTimer(&timer);
}

void VRWidget::AddRenderable(const char* name, void* r)
{
	renderers[name] = (Renderable*)r;
	((Renderable*)r)->SetAllRenderable(&renderers);
	//((Renderable*)r)->SetActor(this);
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

	ctx = std::make_unique<osvr::clientkit::ClientContext>("com.osvr.example.SDLOpenGL");
	display = std::make_unique<osvr::clientkit::DisplayConfig>(*ctx.get());
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
	GLfloat modelview[16];
	GLfloat projection[16];
	mainGLWidget->GetModelview(modelview);
	mainGLWidget->GetProjection(projection);
	makeCurrent();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	display->forEachEye([](osvr::clientkit::Eye eye) {

		/// Try retrieving the view matrix (based on eye pose) from OSVR
		double viewMat[OSVR_MATRIX_SIZE];
		eye.getViewMatrix(OSVR_MATRIX_COLMAJOR | OSVR_MATRIX_COLVECTORS,
			viewMat);
		/// Initialize the ModelView transform with the view matrix we
		/// received
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glMultMatrixd(viewMat);

		/// For each display surface seen by the given eye of the given
		/// viewer...
		eye.forEachSurface([](osvr::clientkit::Surface surface) {
			auto viewport = surface.getRelativeViewport();
			glViewport(static_cast<GLint>(viewport.left),
				static_cast<GLint>(viewport.bottom),
				static_cast<GLsizei>(viewport.width),
				static_cast<GLsizei>(viewport.height));

			/// Set the OpenGL projection matrix based on the one we
			/// computed.
			double zNear = 0.1;
			double zFar = 100;
			double projMat[OSVR_MATRIX_SIZE];
			surface.getProjectionMatrix(
				zNear, zFar, OSVR_MATRIX_COLMAJOR | OSVR_MATRIX_COLVECTORS |
				OSVR_MATRIX_SIGNEDZ | OSVR_MATRIX_RHINPUT,
				projMat);

			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glMultMatrixd(projMat);

			/// Set the matrix mode to ModelView, so render code doesn't
			/// mess with the projection matrix on accident.
			glMatrixMode(GL_MODELVIEW);

			/// Call out to render our scene.
			//renderScene();
		});
	});



	glViewport(0, 0, width / 2,height);
	//((GlyphRenderable*)GetRenderable("glyph"))->SetDispalceOn(false);
	for (auto renderer : renderers)
		renderer.second->draw(modelview, projection);
	glViewport(width / 2, 0, width / 2, height);
	for (auto renderer : renderers)
		renderer.second->draw(modelview, projection);
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

Renderable* VRWidget::GetRenderable(const char* name)
{
	if (renderers.find(name) == renderers.end()) {
		std::cout << "No renderer named : " << name << std::endl;
		exit(1);
	}
	return renderers[name];
}

void VRWidget::UpdateGL()
{
	update();
}
