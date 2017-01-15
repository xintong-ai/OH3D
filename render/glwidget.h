#ifndef GL_WIDGET_H
#define GL_WIDGET_H

#define ENABLE_TIMER 1

#include <QtWidgets>
#include <QVector3D>
#include <QMatrix4x4>
#include <QOpenGLWidget>
#include <vector_types.h>
#include <vector_functions.h>
#include <memory>
#include <CMakeConfig.h>

enum INTERACT_MODE{
	MOVE_LENS,
	MODIFY_LENS_FOCUS_SIZE,
	MODIFY_LENS_TRANSITION_SIZE,
	MODIFY_LENS_DEPTH,
	MODIFY_LENS_TWO_FINGERS,
	TRANSFORMATION,
	ADDING_LENS,

	CHANGING_FORCE //currently only used for Leap
};

enum DEFORM_MODEL{
	OBJECT_SPACE,
	SCREEN_SPACE,
};


class StopWatchInterface;
class Renderable;
class GLMatrixManager;
class Processor;
class Interactor;

#ifdef USE_OSVR
class VRWidget;
#endif

class GLWidget : public QOpenGLWidget, public QOpenGLFunctions
{
    Q_OBJECT
public:

    explicit GLWidget(std::shared_ptr<GLMatrixManager> _matrixMgr,
		QWidget *parent = 0);
    explicit GLWidget(QWidget *parent = 0);
	~GLWidget();

	GLuint framebuffer, renderbuffer[2];
	int xMouse, yMouse;

	void AddRenderable(const char* name, void* r);
	void AddProcessor(const char* name, void* r); //the processors can be moved to DeformGlWidget in the future. or combine DeformGlWidget with glwidget
	void AddInteractor(const char* name, void* r);

	QSize minimumSizeHint() const Q_DECL_OVERRIDE;
	QSize sizeHint() const Q_DECL_OVERRIDE;
    void GetWindowSize(int &w, int &h) {w = width; h = height;}
	int2 GetWindowSize() { return make_int2(width, height); }

	float3 DataCenter();	
	void GetPosRange(float3 &pmin, float3 &pmax);

	void UpdateGL();

	void GetModelview(float* m);// { for (int i = 0; i < 16; i++) m[i] = modelview[i]; }
	void GetProjection(float* m);// { for (int i = 0; i < 16; i++) m[i] = projection[i]; }

	INTERACT_MODE GetInteractMode(){ return interactMode; }
	void SetInteractMode(INTERACT_MODE v);// { interactMode = v; std::cout << "Set INTERACT_MODE: " << interactMode << std::endl; }
	
#ifdef USE_LEAP
	float blendOthers = false; //when draw the hand cursor in 3D, blend the particles when necessary
#endif

#ifdef USE_OSVR
	void SetVRWidget(VRWidget* _vrWidget){ vrWidget = _vrWidget; }
#endif

protected:
	uint width = 750, height = 900;

	std::shared_ptr<GLMatrixManager> matrixMgr;

	std::map<std::string, Renderable*> renderers;
	std::map<std::string, Processor*> processors;
	std::map<std::string, Interactor*> interactors;

    virtual void initializeGL() Q_DECL_OVERRIDE;
    virtual void paintGL() Q_DECL_OVERRIDE;
    virtual void resizeGL(int width, int height) Q_DECL_OVERRIDE;
    virtual void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    virtual void mouseReleaseEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    virtual void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    virtual void wheelEvent(QWheelEvent * event) Q_DECL_OVERRIDE;
    virtual void keyPressEvent(QKeyEvent * event) Q_DECL_OVERRIDE;
	virtual bool event(QEvent *event) Q_DECL_OVERRIDE;

	virtual bool TouchBeginEvent(QTouchEvent *event){ return false; }
	virtual bool TouchUpdateEvent(QTouchEvent *event){ return false; }
	virtual void pinchTriggered(QPinchGesture *gesture);

	QPoint pixelPosToGLPos(const QPoint& p);
	QPoint pixelPosToGLPos(const QPointF& p);
	QPointF pixelPosToViewPos(const QPointF& p);
	QPointF prevPos;//previous mouse position
	

private:
	INTERACT_MODE interactMode = INTERACT_MODE::TRANSFORMATION;
	
	bool initialized = false;
	bool pinching = false;
	//mark whether there is any pinching gesture in this sequence of gestures.
	// in order to prevent rotation if pinching is finished while one finger is still on the touch screen.
	bool pinched = false;

#ifdef USE_OSVR
	VRWidget* vrWidget = nullptr;
#endif
	
	bool gestureEvent(QGestureEvent *event);
	bool TouchEndEvent(QTouchEvent *event); 

		
    /****timing****/
    StopWatchInterface *timer = 0;
    int m_frame;
    int fpsCount = 0;        // FPS count for averaging
    int fpsLimit = 64;        // FPS limit for sampling
    unsigned int frameCount = 0;
	void computeFPS();
	void TimerStart();
	void TimerEnd();

signals:
	void SignalPaintGL();
};

#endif //GL_WIDGET_H
