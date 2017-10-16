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
	
	float colorTableTomato[5][4] = {
		0, 0.0, 0.0, 0,
		30, 51 , 8, 0 ,
		42 , 255 , 99, 71 ,
		68 , 255 , 212 , 204.0,
		1.0, 1.0, 1.0, 1.0
	};

	int sliceOffset = z*w * h;
	QImage image(w*multiplier, h*multiplier, QImage::Format_RGB32);
	for (int i = 0; i<w; ++i) {
		for (int j = 0; j<h; ++j) {
			if (labelVolLocal && labelVolLocal[sliceOffset + w*j + i]){
				for (int ii = 0; ii < multiplier; ii++){
					for (int jj = 0; jj < multiplier; jj++){
						image.setPixel(i*multiplier + ii, (h - 1 - j)*multiplier + jj, 256 * 256 * 0 + 256 * 255 + 255);
					}
				}
			}
			else{
				int vv = 255 * inputVolume->values[sliceOffset + j*w + i];
				for (int ii = 0; ii < multiplier; ii++){
					for (int jj = 0; jj < multiplier; jj++){
						image.setPixel(i*multiplier + ii, (h - 1 - j)*multiplier + jj, 256 * 256 * vv + 256 * vv + vv);
						////for Tomato data
						//int pos = 0;
						//if (vv <= 30){
						//	image.setPixel(i*multiplier + ii, (h-1-j)*multiplier + jj, 0);
						//}
						//else if(vv<=42){
						//	pos = 1;
						//}
						//else if (vv <= 68){
						//	pos = 2;
						//}
						//else{
						//	pos = 3;
						//}
						//if (pos > 0){
						//	float ratio = (vv - colorTableTomato[pos][0]) / (colorTableTomato[pos + 1][0] - colorTableTomato[pos][0]);
						//	
						//	int3 val = make_int3(
						//		ratio*(colorTableTomato[pos + 1][1] - colorTableTomato[pos][1]) + colorTableTomato[pos][1],
						//		ratio*(colorTableTomato[pos + 1][2] - colorTableTomato[pos][2]) + colorTableTomato[pos][2],
						//		ratio*(colorTableTomato[pos + 1][3] - colorTableTomato[pos][3]) + colorTableTomato[pos][3]);
						//	image.setPixel(i*multiplier + ii, (h-1-j)*multiplier + jj, 256 * 256 * val.x + 256 * val.y + val.z);
						//}
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


	if (needSaveImage){
		SYSTEMTIME st;
		GetSystemTime(&st);
		std::string fname = "2dScreenShot_" + std::to_string(st.wMinute) + std::to_string(st.wSecond) + std::to_string(st.wMilliseconds) + ".png";
		std::cout << "saving screen shot to file: " << fname << std::endl;
		image.save(fname.c_str());
		needSaveImage = false;
		std::cout << "finish saving image" << std::endl;
	}
}

void Helper::featureGrowing()
{
	if (!labelVolLocal){
		std::cout << "labelVolLocal not set!!" << std::endl;
		return;
	}

	if (!valSet){
		return;
	}

	int3 sixNb[6] = { make_int3(-1, 0, 0), make_int3(0, -1, 0), make_int3(0, 0, -1),
		make_int3(1, 0, 0), make_int3(0, 1, 0), make_int3(0, 0, 1) };

	int3 size = inputVolume->size;
	for (int k = 0; k < size.z; k++){
		for (int j = 0; j < size.y; j++){
			for (int i = 0; i < size.x; i++){
				//if (abs(val - inputVolume->values[k*w*h + j*w + i]) >= 0.1){
				//	continue;
				//}
				if (inputVolume->values[k*w*h + j*w + i] - val >= 0.1 || inputVolume->values[k*w*h + j*w + i] < 0.1){ //here 0.1 is the transfer function cut off value
					continue;
				}


				bool notFound = true;

				for (int tt = 0; tt < 6 && notFound; tt++){
					int xq = i + sixNb[tt].x, yq = j + sixNb[tt].y, zq = k + sixNb[tt].z;
					if (xq >= 0 && xq < size.x && yq >= 0 && yq < size.y && zq >= 0 && zq < size.z){
						if (labelVolLocal[zq*w*h + yq*w + xq] == 1){
							labelVolLocal[k*w*h + j*w + i] = 2;
							notFound = true;
						}
					}
				}
			}
		}
	}
	for (int i = 0; i < size.z*size.y*size.x; i++){
		if (labelVolLocal[i] == 2)
			labelVolLocal[i] = 1;
	}
	std::cout << "region growing for one voxel"<< std::endl;

}

void Helper::brushPoint(int _x, int _y)
{
	if (!labelVolLocal){
		std::cout << "labelVolLocal not set!!" << std::endl;
		return;
	}
	const int cons = 3;
	int x = _x / multiplier, y = h - 1 - _y / multiplier;

	if (!valSet){
		val = inputVolume->values[z*w*h + y*w + x];
		valSet = true;
	}

	int sliceOffset = z*w * h;
	for (int xx = x - cons; xx <= x + cons; xx++){
		for (int yy = y - cons; yy <= y + cons; yy++){
			if (xx >= 0 && xx < w && yy >= 0 && yy < h && abs(val - inputVolume->values[z*w*h + yy*w + xx])<0.05){
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
		helper->brushPoint(point.x(), point.y());
	}
}

void GLWidgetQtDrawing::mouseMoveEvent(QMouseEvent *event)
{
	if (pressed) {
		QPoint point = event->pos();
		//std::cout << point.x() << " " << point.y() << std::endl;
		helper->brushPoint(point.x(), point.y());
	}
}

void GLWidgetQtDrawing::mouseReleaseEvent(QMouseEvent *event)
{
	pressed = false;
}
