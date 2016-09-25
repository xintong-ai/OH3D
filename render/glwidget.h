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
	//	DRAG_LENS_EDGE,
	//	DRAG_LENS_TWO_ENDS,
	MOVE_LENS,
	MODIFY_LENS_FOCUS_SIZE,
	MODIFY_LENS_TRANSITION_SIZE,
	MODIFY_LENS_DEPTH,
	MODIFY_LENS_TWO_FINGERS,
	TRANSFORMATION,
	ADDING_LENS,
	//CUT_LINE,
//	NO_TRANSFORMATION,
	//ADD_NODE,
	//MODIFY_LENS,
	//DRAW_ELLIPSE,
};

enum DEFORM_MODEL{
	OBJECT_SPACE,
	SCREEN_SPACE,
};


class StopWatchInterface;
class Renderable;
class GLMatrixManager;

#ifdef USE_OSVR
class VRWidget;
#endif

class GLWidget : public QOpenGLWidget, public QOpenGLFunctions
{
    Q_OBJECT
public:

	GLuint framebuffer, renderbuffer[2];
	int xMouse, yMouse;
    explicit GLWidget(std::shared_ptr<GLMatrixManager> _matrixMgr,
		QWidget *parent = 0);


    explicit GLWidget(QWidget *parent = 0);

	~GLWidget();

    QSize minimumSizeHint() const Q_DECL_OVERRIDE;
    QSize sizeHint() const Q_DECL_OVERRIDE;

	void AddRenderable(const char* name, void* r);

	Renderable* GetRenderable(const char* name);

    void GetWindowSize(int &w, int &h) {w = width; h = height;}

	int2 GetWindowSize() { return make_int2(width, height); }

	float3 DataCenter();
	
	void GetPosRange(float3 &pmin, float3 &pmax);

	void UpdateGL();

	void SetInteractMode(INTERACT_MODE v);// { interactMode = v; std::cout << "Set INTERACT_MODE: " << interactMode << std::endl; }

	void GetModelview(float* m);// { for (int i = 0; i < 16; i++) m[i] = modelview[i]; }
	void GetProjection(float* m);// { for (int i = 0; i < 16; i++) m[i] = projection[i]; }
	
	INTERACT_MODE GetInteractMode(){ return interactMode; }

#ifdef USE_OSVR
	void SetVRWidget(VRWidget* _vrWidget){ vrWidget = _vrWidget; }
#endif


protected:
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

	uint width = 750, height = 900;
	std::shared_ptr<GLMatrixManager> matrixMgr;

	std::map<std::string, Renderable*> renderers;


private:
#ifdef USE_OSVR
	VRWidget* vrWidget = nullptr;
#endif

    void computeFPS();

    void TimerStart();

    void TimerEnd();


    QPointF pixelPosToViewPos(const QPointF& p);



	bool gestureEvent(QGestureEvent *event);

	bool TouchEndEvent(QTouchEvent *event); 

		/*****view*****/

    QPointF prevPos;//previous mouse position

	
    /****timing****/
    StopWatchInterface *timer = 0;
    int m_frame;
    int fpsCount = 0;        // FPS count for averaging
    int fpsLimit = 64;        // FPS limit for sampling
    unsigned int frameCount = 0;


    bool initialized = false;
	bool pinching = false;
	//mark whether there is any pinching gesture in this sequence of gestures.
	// in order to prevent rotation if pinching is finished while one finger is still on the touch screen.
	bool pinched = false;	


	//transformation states
	QVector3D transVec = QVector3D(0.0f, 0.0f, -5.0f);//move it towards the front of the camera
	QMatrix4x4 transRot;
	float transScale = 1;
	float currentTransScale = 1;

	float3 dataMin = make_float3(0, 0, 0);
	float3 dataMax = make_float3(10, 10, 10);

	INTERACT_MODE interactMode = INTERACT_MODE::TRANSFORMATION;

};

#endif //GL_WIDGET_H
