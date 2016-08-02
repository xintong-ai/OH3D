#include "glwidget.h"

class DeformGLWidget : public GLWidget
{
public:
	bool isPicking = false;
	int pickID = -1;
	void GetDepthRange(float2& v){ v = depthRange; }

	void SetDeformModel(DEFORM_MODEL v) { deformModel = v; }
	DEFORM_MODEL GetDeformModel() { return deformModel; }

	explicit DeformGLWidget(std::shared_ptr<GLMatrixManager> _matrixMgr,
		QWidget *parent = 0) : GLWidget(_matrixMgr, parent)
	{}

protected:
	virtual void mouseReleaseEvent(QMouseEvent *event) Q_DECL_OVERRIDE;
	virtual void wheelEvent(QWheelEvent * event) Q_DECL_OVERRIDE;
	virtual bool TouchBeginEvent(QTouchEvent *event) override;
	virtual bool TouchUpdateEvent(QTouchEvent *event) override;
	virtual void pinchTriggered(QPinchGesture *gesture) override;

private:
	void UpdateDepthRange();

	//int2 lastPt = make_int2(0,0);

	float2 depthRange;



	DEFORM_MODEL deformModel = DEFORM_MODEL::SCREEN_SPACE; //
	bool insideLens = false;

private slots:
	void animate();
};

