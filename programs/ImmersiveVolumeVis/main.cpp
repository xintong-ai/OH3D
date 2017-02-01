
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

#include "ViewpointEvaluator.h"

using namespace std;

//#define debugoutput


int main(int argc, char **argv)
{
	//using the following statement, 
	//all the OpenGL contexts in this program are shared.
	QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
	QApplication app(argc, argv);
	Window win;

	win.init();
	//win.setFixedSize(1095, 822);



	//std::shared_ptr<ViewpointEvaluator> ve = std::make_shared<ViewpointEvaluator>(win.rcp, win.inputVolume);
	//if (win.useLabel){
	//	ve->setLabel(win.labelVol);
	//}
	//ve->initDownSampledResultVolume(make_int3(40, 40, 40));
	//ve->compute(VPMethod::JS06Sphere);
	//ve->saveResultVol("entro.raw");


	win.show();
	return app.exec();


	return 1;

}
