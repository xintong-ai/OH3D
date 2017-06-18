#include "DeformGLWidget.h"
#include "GLMatrixManager.h"
#include "TransformFunc.h"

DeformGLWidget::DeformGLWidget(std::shared_ptr<GLMatrixManager> _matrixMgr,
	QWidget *parent) : GLWidget(_matrixMgr, parent)
{
	QTimer *aTimer = new QTimer;
	connect(aTimer, SIGNAL(timeout()), SLOT(animate()));
	aTimer->start(30);
}

void DeformGLWidget::animate()
{
	update(); 
}