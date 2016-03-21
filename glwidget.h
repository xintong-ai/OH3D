#ifndef GL_WIDGET_H
#define GL_WIDGET_H

#include <QtWidgets>
#include <QVector3D>
#include <QMatrix4x4>
#include <QOpenGLWidget>
#include <vector_types.h>
#include <vector_functions.h>
#include <memory>

enum INTERACT_MODE{
	//	DRAG_LENS_EDGE,
	//	DRAG_LENS_TWO_ENDS,
	MOVE_LENS,
	MODIFY_LENS_FOCUS_SIZE,
	MODIFY_LENS_TRANSITION_SIZE,
	MODIFY_LENS_DEPTH,
	TRANSFORMATION,
	MODIFYING_LENS,
	//CUT_LINE,
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
class VRWidget;
class GLMatrixManager;

class GLWidget : public QOpenGLWidget, public QOpenGLFunctions
{
    Q_OBJECT
public:

	bool isPicking = false;
	GLuint framebuffer, renderbuffer[2];
	int xMouse, yMouse;
	int pickID = -1;
    explicit GLWidget(std::shared_ptr<GLMatrixManager> _matrixMgr, QWidget *parent = 0);


    explicit GLWidget(QWidget *parent = 0);

	~GLWidget();

    QSize minimumSizeHint() const Q_DECL_OVERRIDE;
    QSize sizeHint() const Q_DECL_OVERRIDE;

	void AddRenderable(const char* name, void* r);

	Renderable* GetRenderable(const char* name);

    void GetWindowSize(int &w, int &h) {w = width; h = height;}

	int2 GetWindowSize() { return make_int2(width, height); }

	float3 DataCenter();


	void UpdateGL();


	INTERACT_MODE GetInteractMode(){ return interactMode; }

	void SetInteractMode(INTERACT_MODE v) { interactMode = v; }

	void GetModelview(float* m);// { for (int i = 0; i < 16; i++) m[i] = modelview[i]; }
	void GetProjection(float* m);// { for (int i = 0; i < 16; i++) m[i] = projection[i]; }

	void SetVRWidget(VRWidget* _vrWidget){ vrWidget = _vrWidget; }

	void GetDepthRange(float2& v){ v = depthRange; }

	void SetDeformModel(DEFORM_MODEL v) { deformModel = v; }
	DEFORM_MODEL GetDeformModel() { return deformModel; }

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

	uint width = 750, height = 900;
private:
	VRWidget* vrWidget = nullptr;
    void computeFPS();

    void TimerStart();

    void TimerEnd();

	void UpdateDepthRange();

    QPointF pixelPosToViewPos(const QPointF& p);

	QPoint pixelPosToGLPos(const QPoint& p);

	bool gestureEvent(QGestureEvent *event);

	bool TouchEvent(QTouchEvent *event);

	void pinchTriggered(QPinchGesture *gesture/*, QPointF center*/);
		/*****view*****/

    QPointF prevPos;//previous mouse position

	INTERACT_MODE interactMode = INTERACT_MODE::TRANSFORMATION;

    /****timing****/
    StopWatchInterface *timer = 0;
    int m_frame;
    int fpsCount = 0;        // FPS count for averaging
    int fpsLimit = 16;        // FPS limit for sampling
    int g_Index = 0;
    unsigned int frameCount = 0;

    std::map<std::string,Renderable*> renderers;

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



	float2 depthRange;

	//int2 lastPt = make_int2(0,0);

	std::shared_ptr<GLMatrixManager> matrixMgr;

	DEFORM_MODEL deformModel = DEFORM_MODEL::OBJECT_SPACE;// DEFORM_MODEL::SCREEN_SPACE; //
	bool insideLens = false;
private slots:
	void animate();

};

#endif //GL_WIDGET_H
