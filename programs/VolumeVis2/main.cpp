
//#include "glextensions.h"
#include <string>
#include <QApplication>
#include <window.h>

int main(int argc, char **argv)
{
	//using the following statement, 
	//all the OpenGL contexts in this program are shared.
	QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
	QApplication app(argc, argv);

	Window win;
	win.show();
	win.init();

	win.setFixedSize(1095, 822);

	return app.exec();
}
