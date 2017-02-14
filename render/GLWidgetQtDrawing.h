#ifndef GL_WIDGET_QTDRAWING_H
#define GL_WIDGET_QTDRAWING_H

#include <QtWidgets>
#include <memory>
#include <vector>

class Volume;

class Helper
{
public:
	Helper();
	int w, h;
	int z;
	unsigned short* labelVolLocal;
	bool valSet = false;

	int multiplier = 1;
public:
	void paint(QPainter *painter, QPaintEvent *event, int elapsed);
	void mousePress(int x, int y);
	void setData(std::shared_ptr<Volume> _v, unsigned short* _l);

private:
	std::shared_ptr<Volume> inputVolume;

	float val;

	QBrush background;
	QBrush circleBrush;
	QFont textFont;
	QPen circlePen;
	QPen textPen;
};


class GLWidgetQtDrawing : public QOpenGLWidget
{
	Q_OBJECT

public:
	GLWidgetQtDrawing(Helper *helper, QWidget *parent);
	
	public slots:
	void animate();

protected:
	void paintEvent(QPaintEvent *event) override;
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	bool pressed = false;

private:
	Helper *helper;
	int elapsed;
};




#endif