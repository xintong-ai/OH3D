#ifndef VR_WIDGET_H
#define VR_WIDGET_H

#include <QtWidgets>
#include <QVector3D>
#include <QMatrix4x4>
#include <QOpenGLWidget>
#include <vector_types.h>
#include <vector_functions.h>
#include <memory>
//enum INTERACT_MODE{
//	//	DRAG_LENS_EDGE,
//	//	DRAG_LENS_TWO_ENDS,
//	LENS,
//	TRANSFORMATION,
//	MODIFYING_LENS,
//	//CUT_LINE,
//	//ADD_NODE,
//	MODIFY_LENS,
//	//DRAW_ELLIPSE,
//};


//class Trackball;
//class Rotation;
class StopWatchInterface;
class Renderable;
class GLWidget;
class GLMatrixManager;
namespace osvr{
	namespace clientkit{
		class ClientContext;
		class DisplayConfig;
	}
}

class VRWidget : public QOpenGLWidget, public QOpenGLFunctions
{
	Q_OBJECT
public:
	explicit VRWidget(std::shared_ptr<GLMatrixManager> _matrixMgr, QWidget *parent = 0);
	~VRWidget();

	QSize minimumSizeHint() const Q_DECL_OVERRIDE;
	QSize sizeHint() const Q_DECL_OVERRIDE;

	void AddRenderable(const char* name, void* r);

	void GetWindowSize(int &w, int &h) { w = width; h = height; }

	int2 GetWindowSize() { return make_int2(width, height); }

	//void SetVol(int3 dim);

	//void SetVol(float3 posMin, float3 posMax);
	//void GetVol(float3 &posMin, float3 &posMax){ posMin = dataMin; posMax = dataMax; }
	//float3 DataCenter();
	void UpdateGL();

protected:
	virtual void initializeGL() Q_DECL_OVERRIDE;
	virtual void paintGL() Q_DECL_OVERRIDE;
	virtual void resizeGL(int width, int height) Q_DECL_OVERRIDE;
	//virtual void mousePressEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
	//virtual void mouseReleaseEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
	//virtual void mouseMoveEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
	//virtual void wheelEvent(QWheelEvent * event) Q_DECL_OVERRIDE;
	virtual void keyPressEvent(QKeyEvent * event) Q_DECL_OVERRIDE;
	//virtual bool event(QEvent *event) Q_DECL_OVERRIDE;

	uint width = 750, height = 900;


private:
	void computeFPS();

	void TimerStart();

	void TimerEnd();

	/****timing****/
	StopWatchInterface *timer = 0;
	int m_frame;
	int fpsCount = 0;        // FPS count for averaging
	int fpsLimit = 128;        // FPS limit for sampling
	int g_Index = 0;
	unsigned int frameCount = 0;

	std::map<std::string, Renderable*> renderers;

	bool initialized = false;
	std::shared_ptr<GLMatrixManager> matrixMgr;

	std::shared_ptr<osvr::clientkit::ClientContext> ctx;// ("com.osvr.example.SDLOpenGL");
	std::shared_ptr<osvr::clientkit::DisplayConfig> display;// (ctx);

};

#endif //VR_WIDGET_H
