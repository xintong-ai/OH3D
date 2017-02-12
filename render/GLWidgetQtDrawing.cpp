#include "GLWidgetQtDrawing.h"
#include "Volume.h"
#include <QColor>
#include <iostream>

Helper::Helper()
{
	QLinearGradient gradient(QPointF(50, -20), QPointF(80, 20));
	gradient.setColorAt(0.0, Qt::white);
	gradient.setColorAt(1.0, QColor(0xa6, 0xce, 0x39));

	background = QBrush(QColor(64, 32, 64));
	circleBrush = QBrush(gradient);
	circlePen = QPen(Qt::black);
	circlePen.setWidth(1);
	textPen = QPen(Qt::white);
	textFont.setPixelSize(50);
}

void Helper::setData(std::shared_ptr<Volume> _v, unsigned short* l){
	inputVolume = _v;
	z = inputVolume->size.z / 2;
	w = inputVolume->size.x;
	h = inputVolume->size.y;
	labelVolLocal = l;
}

void Helper::paint(QPainter *painter, QPaintEvent *event, int elapsed)
{
	painter->fillRect(event->rect(), background);
	//painter->translate(100, 100);
	painter->save();
	painter->setBrush(circleBrush);
	painter->setPen(circlePen);
	//painter->rotate(elapsed * 0.030);

	int sliceOffset = z*w * h;
	QImage image(w*multiplier, h*multiplier, QImage::Format_RGB32);
	for (int i = 0; i<w; ++i) {
		for (int j = 0; j<h; ++j) {
			if (labelVolLocal[sliceOffset + w*j + i]){
				for (int ii = 0; ii < multiplier; ii++){
					for (int jj = 0; jj < multiplier; jj++){
						image.setPixel(i*multiplier + ii, j*multiplier + jj, 256 * 256 * 255 + 256 * 255 + 0);
					}
				}
			}
			else{
				int vv = 255 * inputVolume->values[sliceOffset + j*w + i];
				for (int ii = 0; ii < multiplier; ii++){
					for (int jj = 0; jj < multiplier; jj++){
						image.setPixel(i*multiplier + ii, j*multiplier + jj, 256 * 256 * vv + 256 * vv + vv);
					}
				}
			}
		}
	}


	/*
	qreal r = elapsed / 1000.0;
	int n = 30;
	for (int i = 0; i < n; ++i) {
		painter->rotate(30);
		qreal factor = (i + r) / n;
		qreal radius = 0 + 120.0 * factor;
		qreal circleRadius = 1 + factor * 20;
		painter->drawEllipse(QRectF(radius, -circleRadius,
			circleRadius * 2, circleRadius * 2));
	}
	*/
	painter->drawImage(QPointF(0,0), image);

	painter->restore();
	painter->setPen(textPen);
	painter->setFont(textFont);
	//painter->drawText(QRect(-50, -50, 100, 100), Qt::AlignCenter, QStringLiteral("Qt"));
}

void Helper::mousePress(int _x, int _y)
{
	const int cons = 3;
	int x = _x / multiplier, y = _y / multiplier;

	if (!valSet){
		val = inputVolume->values[z*w*h + y*w + x];
		valSet = true;
	}

	int sliceOffset = z*w * h;
	for (int xx = x - cons; xx <= x + cons; xx++){
		for (int yy = y - cons; yy <= y + cons; yy++){
			if (xx >= 0 && xx < w && yy >= 0 && yy < h && abs(val - inputVolume->values[z*w*h + yy*w + xx])<0.01){
				labelVolLocal[sliceOffset + w*yy + xx] = 1;
			}
		}
	}

}


void GLWidgetQtDrawing::paintEvent(QPaintEvent *event)
{
	QPainter painter;
	painter.begin(this);
	painter.setRenderHint(QPainter::Antialiasing);
	helper->paint(&painter, event, elapsed);
	painter.end();
}

GLWidgetQtDrawing::GLWidgetQtDrawing(Helper *helper, QWidget *parent)
: QOpenGLWidget(parent), helper(helper)
{
	elapsed = 0;
	//setFixedSize(200, 200);
	setFixedSize(helper->w*helper->multiplier, helper->h*helper->multiplier);
	setAutoFillBackground(false);
}

void GLWidgetQtDrawing::animate()
{
	elapsed = (elapsed + qobject_cast<QTimer*>(sender())->interval()) % 1000;
	update();
}

void GLWidgetQtDrawing::mousePressEvent(QMouseEvent *event)
{
	if (event->button() == Qt::LeftButton) {
		QPoint point = event->pos();
		pressed = true;

//		std::cout << point.x() << " " << point.y() << std::endl;
		helper->mousePress(point.x(), point.y());
	}
}

void GLWidgetQtDrawing::mouseMoveEvent(QMouseEvent *event)
{
	if (pressed) {
		QPoint point = event->pos();
		//std::cout << point.x() << " " << point.y() << std::endl;
		helper->mousePress(point.x(), point.y());
	}
}

void GLWidgetQtDrawing::mouseReleaseEvent(QMouseEvent *event)
{
	pressed = false;
}
