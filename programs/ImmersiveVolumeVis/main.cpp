
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


//	ViewpointEvaluator ve(inputVolume);
//	ve.rcp = RayCastingParameters(1.0, 0.2, 0.7, 0.44, 0.29, 1.25, 512, 0.25f, 1.3, false);
//	int3 sampleSize = make_int3(40, 40, 40);
//	Volume entroVolumeDS;
//	entroVolumeDS.setSize(sampleSize);
//	float * d_r;
//
//	bool useSpherewise = true;
//	bool useVolumewise = false;
//	if (useSpherewise){
//		int numSphereSample = 200;
//		ve.setSpherePoints(numSphereSample);
//
//		cudaMalloc(&d_r, sizeof(float)*numSphereSample);
//		for (int k = 0; k < sampleSize.z; k++){
//			cout << "now doing k = " << k << endl;
//
//			for (int j = 0; j < sampleSize.y; j++){
//				for (int i = 0; i < sampleSize.x; i++){
//					float3 eyeInLocal = make_float3(
//						1.0*(i + 1) / (sampleSize.x + 1)*inputVolume->size.x,
//						1.0*(j + 1) / (sampleSize.y + 1)*inputVolume->size.y,
//						1.0*(k + 1) / (sampleSize.z + 1)*inputVolume->size.z
//						);
//					
//					eyeInLocal = make_float3(
//						((1.0*(i + 1) / (sampleSize.x + 1) - 0.5) / 1.1 + 0.5)*inputVolume->size.x,
//						((1.0*(j + 1) / (sampleSize.y + 1) - 0.5) / 1.5 + 0.5)*inputVolume->size.y,
//						((1.0*(k + 1) / (sampleSize.z + 1) - 0.5) / 1.5 + 0.5)*inputVolume->size.z
//						);
//					//cout << "eye " << eyeInLocal.x << " " << eyeInLocal.y << " " << eyeInLocal.z << endl;
//					entroVolumeDS.values[k*sampleSize.y*sampleSize.x + j*sampleSize.x + i] = ve.computeSpherewhiseEntropy(eyeInLocal, d_r);
//					//cout << entroVolumeDS.values[k*sampleSize.y*sampleSize.x + j*sampleSize.x + i] << endl;
//				}
//			}
//		}
//	}
//	if (useVolumewise){
//		int ll = inputVolume->size.x*inputVolume->size.y*inputVolume->size.z;
//		cudaMalloc(&d_r, sizeof(float)*ll);
//		for (int k = 0; k < sampleSize.z; k++){
//			cout << "now doing k = " << k << endl;
//			for (int j = 0; j < sampleSize.y; j++){
//				for (int i = 0; i < sampleSize.x; i++){
//					float3 eyeInLocal = make_float3(
//						1.0*(i + 1) / (sampleSize.x + 1)*inputVolume->size.x,
//						1.0*(j + 1) / (sampleSize.y + 1)*inputVolume->size.y,
//						1.0*(k + 1) / (sampleSize.z + 1)*inputVolume->size.z
//						);
//					cout << "eye " << eyeInLocal.x << " " << eyeInLocal.y << " " << eyeInLocal.z << endl;
//					entroVolumeDS.values[k*sampleSize.y*sampleSize.x + j*sampleSize.x + i] = ve.computeVolumewhiseEntropy(eyeInLocal, d_r);
//					cout << entroVolumeDS.values[k*sampleSize.y*sampleSize.x + j*sampleSize.x + i] << endl;
//				}
//			}
//		}
//	}
//	entroVolumeDS.saveRawToFile("entro.raw");
//#ifdef debugoutput
//	float * visib = new float[inputVolume->size.x*inputVolume->size.y*inputVolume->size.z];
//	for (int i = 0; i < inputVolume->size.x*inputVolume->size.y*inputVolume->size.z; i++){
//		visib[i] = i;
//	}
//	checkCudaErrors(cudaMemcpy(visib, d_r, sizeof(float)*inputVolume->size.x*inputVolume->size.y*inputVolume->size.z, cudaMemcpyDeviceToHost));
//	FILE * fp = fopen("visib.raw", "wb");
//	fwrite(visib, sizeof(float), ll, fp);
//	fclose(fp);
//	delete visib;
//#endif
//	cudaFree(d_r);



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
