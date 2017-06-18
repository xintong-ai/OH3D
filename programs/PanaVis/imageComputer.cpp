#include <iostream>
#include "ImageComputer.h"
#include "GLMatrixManager.h"
#include <sstream>
#include <iomanip>

#include "VolumeRenderableCUDAKernel.h"
//#include "TransformFunc.h"
#include "myDefineRayCasting.h"

using namespace std;

ImageComputer::ImageComputer(std::shared_ptr<GLMatrixManager> m)
{
	matrixMgr = m; 

	cudaMalloc(&d_output, sizeof(uint)* winWidth*winHeight);
	output = new uint[winWidth*winHeight];

	//construct view points
	float3 s = make_float3(67, 74, 107), e = make_float3(67, 137, 107);
	for (int i = 0; i < N; i++){
		viewpoints.push_back(i*1.0 / (N - 1)*(e - s) + s);
	}
};

ImageComputer::~ImageComputer()
{
	delete[] output;
	cudaFree(d_output);
};


void ImageComputer::compute(int3 volumeSize, float3 spacing, std::shared_ptr<RayCastingParameters> rcp)
{
	for (int i = 0; i < N; i++){
		cout << "generating image " << i << endl;

		matrixMgr->moveEyeInLocalByModeMat(viewpoints[counterL]);

		float modelview[16];
		float projection[16];
		matrixMgr->GetModelViewMatrix(modelview);
		matrixMgr->GetProjection(projection);  //note!! this is related with winWidth/winHeight



		QMatrix4x4 q_modelview = QMatrix4x4(modelview).transposed();
		QMatrix4x4 q_invMV = q_modelview.inverted();
		QVector4D q_eye4 = q_invMV.map(QVector4D(0, 0, 0, 1));
		float3 eyeInLocal = make_float3(q_eye4[0], q_eye4[1], q_eye4[2]);

		QMatrix4x4 q_projection = QMatrix4x4(projection).transposed();
		QMatrix4x4 q_mvp = q_projection*q_modelview;
		QMatrix4x4 q_invMVP = q_mvp.inverted();

		float MVMatrix[16];
		float MVPMatrix[16];
		float invMVMatrix[16];
		float invMVPMatrix[16];
		float NMatrix[9];

		q_invMV.copyDataTo(invMVMatrix);
		q_invMVP.copyDataTo(invMVPMatrix); //copyDataTo() automatically copy in row-major order
		q_mvp.copyDataTo(MVPMatrix);
		q_modelview.copyDataTo(MVMatrix);
		q_modelview.normalMatrix().copyDataTo(NMatrix);
		VolumeRender_setConstants(MVMatrix, MVPMatrix, invMVMatrix, invMVPMatrix, NMatrix, &spacing, rcp.get());

		/*
		//assume all related with volume have been well set
		if (volume != 0){
		VolumeRender_computeGradient(&(volume->volumeCuda), &volumeCUDAGradient);
		VolumeRender_setGradient(&volumeCUDAGradient);
		VolumeRender_setVolume(&(volume->volumeCuda));
		}
		else {
		std::cout << "data not well set for volume renderable" << std::endl;
		exit(0);
		}
		*/


		cudaMemset(d_output, 0, sizeof(uint)*winWidth*winHeight);

		//VolumeRender_render(d_output, winWidth, winHeight, matrixMgr->getEyeInLocal(), volumeSize);
		OmniVolumeRender_render(d_output, winWidth, winHeight, matrixMgr->getEyeInLocal(), volumeSize);

		cudaMemcpy(output, d_output, sizeof(uint)*winWidth*winHeight, cudaMemcpyDeviceToHost);
		saveImage(output, winWidth, winHeight);
	}
}

void ImageComputer::saveImage(uint *output, int w, int h)
{
	uchar* pixel = (uchar*)output;




	std::stringstream ss;
	ss << std::setw(5) << std::setfill('0') << counterL;
	std::string fname = "omni_" + ss.str() +".png";
	counterL++;
	std::cout << "saving screen shot to file: " << fname << std::endl;

	std::cout << "w " << w << " h " << h << std::endl;

	QImage image(w, h, QImage::Format_RGB32);
	for (int i = 0; i<w; ++i) {
		for (int j = 0; j<h; ++j) {
			//int jinv = h - 1 - j;
			int jinv = j; //in the d_OmniVolumeRender_preint() function, the j is already positioned correctly
			int ind = (i + j*w) * 4;
			image.setPixel(i, jinv, 256 * 256 * pixel[ind] + 256 * pixel[ind + 1] + pixel[ind + 2]);
		}
	}

	image.save(fname.c_str());

	std::cout << "finish saving image" << std::endl;

}