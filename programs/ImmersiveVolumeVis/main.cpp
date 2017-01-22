
#include <string>
#include <iostream>
#include <QApplication>
#include <window.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "Volume.h"
#include "DataMgr.h"
#include "VecReader.h"
#include "RawVolumeReader.h"
#include "GLMatrixManager.h"
#include "VolumeRenderableCUDA.h"
#include "VolumeRenderableCUDAKernel.h"

#include "ViewpointEvaluator.h"

using namespace std;

//#define debugoutput


int main(int argc, char **argv)
{
	std::shared_ptr<DataMgr> dataMgr;
	dataMgr = std::make_shared<DataMgr>();
	const std::string dataPath = dataMgr->GetConfig("VOLUME_DATA_PATH");

	int3 dims;
	float3 spacing;
	if (std::string(dataPath).find("MGHT2") != std::string::npos){
		dims = make_int3(320, 320, 256);
		spacing = make_float3(0.7, 0.7, 0.7);
	}
	else if (std::string(dataPath).find("MGHT1") != std::string::npos){
		dims = make_int3(256, 256, 176);
		spacing = make_float3(1.0, 1.0, 1.0);
	}
	else if (std::string(dataPath).find("nek128") != std::string::npos){
		dims = make_int3(128, 128, 128);
		spacing = make_float3(2, 2, 2); //to fit the streamline of nek256
	}
	else if (std::string(dataPath).find("nek256") != std::string::npos){
		dims = make_int3(256, 256, 256);
		spacing = make_float3(1, 1, 1);
	}
	else if (std::string(dataPath).find("cthead") != std::string::npos){
		dims = make_int3(208, 256, 225);
		spacing = make_float3(1, 1, 1);
	}
	else if (std::string(dataPath).find("brat") != std::string::npos){
		dims = make_int3(160, 216, 176);
		spacing = make_float3(1, 1, 1);
	}
	else{
		std::cout << "volume data name not recognized" << std::endl;
		exit(0);
	}
	std::shared_ptr<Volume> inputVolume = std::make_shared<Volume>();

	if (std::string(dataPath).find(".vec") != std::string::npos){
		std::shared_ptr<VecReader> reader;
		reader = std::make_shared<VecReader>(dataPath.c_str());
		reader->OutputToVolumeByNormalizedVecMag(inputVolume);
		//reader->OutputToVolumeByNormalizedVecDownSample(inputVolume,2);
		//reader->OutputToVolumeByNormalizedVecUpSample(inputVolume, 2);
		//reader->OutputToVolumeByNormalizedVecMagWithPadding(inputVolume,10);

		reader.reset();
	}
	else{
		std::shared_ptr<RawVolumeReader> reader;
		reader = std::make_shared<RawVolumeReader>(dataPath.c_str(), dims);
		reader->OutputToVolumeByNormalizedValue(inputVolume);
		reader.reset();
	}
	inputVolume->spacing = spacing;
	inputVolume->initVolumeCuda();



	std::shared_ptr<ViewpointEvaluator> ve = std::make_shared<ViewpointEvaluator>(inputVolume);
	//ve->rcp = RayCastingParameters(1.0, 0.2, 0.7, 0.44, 0.29, 1.25, 512, 0.25f, 1.3, false);
	//ve->rcp = RayCastingParameters(1.0, 0.2, 0.7, 0.44, 0.21, 1.25, 512, 0.25f, 1.3, false);
	ve->initDownSampledResultVolume(make_int3(40, 40, 40));
	ve->compute(VPMethod::JS06Sphere);
	ve->saveResultVol("entro.raw");


	//using the following statement, 
	//all the OpenGL contexts in this program are shared.
	QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
	QApplication app(argc, argv);
	Window win(inputVolume);
	win.show();
	win.init();
	//win.setFixedSize(1095, 822);
	return app.exec();


	return 1;

}
