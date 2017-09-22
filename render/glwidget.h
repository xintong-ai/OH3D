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
	OPERATE_MATRIX,

	MOVE_LENS,
	MODIFY_LENS_FOCUS_SIZE,
	MODIFY_LENS_TRANSITION_SIZE,
	MODIFY_LENS_DEPTH,
	ADDING_LENS,

	CHANGING_FORCE, //currently only used for Leap
	PANNING_MATRIX, //currently only used for Leap

	UNDER_TOUCH, //since some touch screen operations are also treated as mouse clicking, currently use this one to prevent confusing operation
	//may need to introduce more to also distinguish between different touch interactors in the future

	OTHERS
};

enum TOUCH_INTERACT_MODE{
	TOUCH_NOT_START,
	TOUCH_OPERATE_MATRIX,
	TOUCH_MOVE_LENS,
	TOUCH_MODIFY_LENS_FOCUS_SIZE,
	TOUCH_MODIFY_LENS_TRANSITION_SIZE,
	TOUCH_MODIFY_LENS_DEPTH,
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
class TouchInteractor;

#ifdef USE_OSVR
class VRWidget;
#endif

class TouchInteractor;

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
	void GetDepthRange(float2 &dr);

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

#ifdef USE_TOUCHSCREEN
	void AddTouchInteractor(const char* name, void* r);
	
	TOUCH_INTERACT_MODE GetTouchInteractMode(){ return touchInteractMode; }
	void SetTouchInteractMode(TOUCH_INTERACT_MODE v) { touchInteractMode = v;  };
#endif

	std::shared_ptr<GLMatrixManager> matrixMgr;

	void saveCurrentImage(){
		needSaveImage = true;
	};
	
	QPoint pixelPosToGLPos(const QPoint& p);
	QPoint pixelPosToGLPos(const QPointF& p);
	QPointF pixelPosToViewPos(const QPointF& p);

protected:
	uint width = 750, height = 900;
	
	std::map<std::string, Renderable*> renderers;
	std::map<std::string, Processor*> processors;
	std::map<std::string, Interactor*> interactors;
	std::map<std::string, Interactor*> matrixInteractors;  
	//interactors and matrixInteractors should be combined together in the future
	//the limited benefits to separate them include that we can make it a must to have at least one matrixInteractor,
	//and to continue using the current strategy of 'interactMode'
	
	


    virtual void initializeGL() Q_DECL_OVERRIDE;
    virtual void paintGL() Q_DECL_OVERRIDE;
    virtual void resizeGL(int width, int height) Q_DECL_OVERRIDE;
    virtual void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    virtual void mouseReleaseEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    virtual void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
    virtual void wheelEvent(QWheelEvent * event) Q_DECL_OVERRIDE;
    virtual void keyPressEvent(QKeyEvent * event) Q_DECL_OVERRIDE;

#ifdef USE_TOUCHSCREEN
	std::map<std::string, TouchInteractor*> touchInteractors;
	bool TouchBeginEvent(QTouchEvent *event);
	bool TouchUpdateEvent(QTouchEvent *event);
	bool TouchEndEvent(QTouchEvent *event);
	void pinchTriggered(QPinchGesture *gesture);
	bool gestureEvent(QGestureEvent *event) { return false; };

	virtual bool event(QEvent *event) Q_DECL_OVERRIDE;

	TOUCH_INTERACT_MODE touchInteractMode = TOUCH_INTERACT_MODE::TOUCH_OPERATE_MATRIX;

	bool pinching = false;
	//mark whether there is any pinching gesture in this sequence of gestures.
	// in order to prevent rotation if pinching is finished while one finger is still on the touch screen.
#endif
	
	QPointF prevPos;//previous mouse position

private:

	bool needSaveImage = false;
	GLuint screenTex = 0;

	INTERACT_MODE interactMode = INTERACT_MODE::OPERATE_MATRIX;
	
	bool initialized = false;

#ifdef USE_OSVR
	VRWidget* vrWidget = nullptr;
#endif
	

    /****timing****/
    StopWatchInterface *timer = 0; //stop and restart at every call of paintGL
    int fpsCount = 0;
    int fpsLimit = 64;
	void TimerStart();
	void TimerEnd();

	StopWatchInterface *timerOverall = 0; //no intermedieta restart
	int fpsCountOverall = 0;

signals:
	void SignalPaintGL();
};

#endif //GL_WIDGET_H
