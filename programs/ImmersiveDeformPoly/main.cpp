#include <string>
#include <iostream>
#include <QApplication>
#include <window.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "Volume.h"

#include "GLMatrixManager.h"
#include "VolumeRenderableCUDA.h"
#include "VolumeRenderableCUDAKernel.h"


using namespace std;

int main(int argc, char **argv)
{
	//using the following statement, 
	//all the OpenGL contexts in this program are shared.
	QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
	QApplication app(argc, argv);
	Window win;

	win.init();
	win.show();
	return app.exec();


	return 1;

}
