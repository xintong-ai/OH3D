#include "glwidget.h"

class DeformGLWidget : public GLWidget
{
	Q_OBJECT
public:
	bool isPicking = false;
	int pickID = -1;

	void SetDeformModel(DEFORM_MODEL v) { deformModel = v; }
	DEFORM_MODEL GetDeformModel() { return deformModel; }

	explicit DeformGLWidget(std::shared_ptr<GLMatrixManager> _matrixMgr,
		QWidget *parent = 0) ;

protected:
	virtual bool TouchBeginEvent(QTouchEvent *event) override;
	virtual bool TouchUpdateEvent(QTouchEvent *event) override;
	virtual void pinchTriggered(QPinchGesture *gesture) override;

private:
	void UpdateDepthRange();


	DEFORM_MODEL deformModel = DEFORM_MODEL::SCREEN_SPACE;

private slots:
	void animate();
};

