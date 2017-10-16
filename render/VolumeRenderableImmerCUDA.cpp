#include <time.h>

#include <vector>

#include <QOpenGLVertexArrayObject>
#include <QOpenGLFunctions_1_2>
#include <QOpenGLFunctions_4_3_Core>
// removing the following lines will cause runtime error
#ifdef WIN32
#include <windows.h>
#endif
#define qgl	QOpenGLContext::currentContext()->functions()

#include <helper_math.h>
#include <cuda_gl_interop.h>

#include "DeformGLWidget.h"
#include "VolumeRenderableImmerCUDA.h"
#include "VolumeRenderableCUDAKernel.h"
#include "TransformFunc.h"
#include "ScreenMarker.h"

#include "PositionBasedDeformProcessor.h"
#include "myDefineRayCasting.h"



VolumeRenderableImmerCUDA::VolumeRenderableImmerCUDA(std::shared_ptr<Volume> _volume, std::shared_ptr<VolumeCUDA> _vlabel, std::shared_ptr<PositionBasedDeformProcessor> p)
{
	volume = _volume;
	volumeCUDAGradient.VolumeCUDA_init(_volume->size, (float*)0, 1, 4);
	if (_vlabel != 0){
		VolumeRender_setLabelVolume(_vlabel.get());
	}

	positionBasedDeformProcessor = p;
}

VolumeRenderableImmerCUDA::VolumeRenderableImmerCUDA(std::shared_ptr<Volume> _volume, std::shared_ptr<PositionBasedDeformProcessor> p)
{
	volume = _volume;
	volumeCUDAGradient.VolumeCUDA_init(_volume->size, (float*)0, 1, 4);

	positionBasedDeformProcessor = p;
}

VolumeRenderableImmerCUDA::~VolumeRenderableImmerCUDA()
{
	VolumeRender_deinit();
	deinitTextureAndCudaArrayOfScreen();

	//if (d_transferFunc != 0){
	//	checkCudaErrors(cudaFreeArray(d_transferFunc));
	//}
	//d_transferFunc = 0;

	//cudaDeviceReset();
};

void VolumeRenderableImmerCUDA::init()
{
	VolumeRender_init(); //must be called before setting color table related variables

	updateColorTable();
	updatePreIntTabelNew(rcp->d_transferFunc);

	initTextureAndCudaArrayOfScreen();

	int winWidth, winHeight;
	actor->GetWindowSize(winWidth, winHeight);
}

void VolumeRenderableImmerCUDA::updateColorTableold()
{
	if (colorTable.size() != numColors){
		colorTable.resize(numColors);
	}
	if (rcp->useColor){
		float peakRange = 0.01;
		std::vector<float> peaks;
		peaks.push_back(rcp->transFuncP2);
		peaks.push_back(rcp->transFuncP1);

		for (int i = 0; i < numColors; i++){
			float normi = i*1.0 / (numColors - 1);
			float colRes;

			if (peaks.size() < 2){
				colRes = normi;
			}
			else{
				colRes = clamp((normi - peaks[0]) / (peaks.back() - peaks[0]), 0.0, 1.0);
			}
			float3 col = make_float3(colRes, 0, 1 - colRes);
			float den = 0.00;

			/*
			float peakWidth = 0;
			float deltab = 0.012, cooefb = 80;
			if (peakWidth == 0){
			for (int j = 0; j < peaks.size(); j++){

			float delta = deltab - 0.002 * j, cooef = cooefb * (j + 1);

			float curDen = cooef * exp(-(peaks[j] - normi)*(peaks[j] - normi) / 2 / delta / delta);
			curDen = max(curDen - 0.1, 0);
			den = max(den, curDen);
			}
			}
			else{
			for (int j = 0; j < peaks.size(); j++){
			float delta = deltab / (j + 1), cooef = cooefb * (j + 1);

			float xv;
			if(abs(normi - peaks[j]) < peakWidth){
			xv = 0;
			}
			else{
			xv = fmin(abs(peaks[j] + peakWidth - normi), abs(peaks[j] - peakWidth - normi));
			}
			float curDen = cooef * exp(-xv*xv / 2 / delta / delta);
			den = max(den, curDen);
			}
			}*/

			float peakWidth = 0.01;
			float deltab = 0.0022, cooefb = 40;
			//exp(-0.01*0.01/2/0.0025/0.0025)*5 = 0.00167731313
			//exp(-0.01*0.01/2/0.002/0.002)*10 = 0.00003726653
			//float v1 = 0.00167731313, v2 = 0.00003726653;
			float v1 = cooefb * exp(-peakWidth * peakWidth / 2 / deltab / deltab);//0.00130480841
			float v2 = cooefb * 2 * exp(-peakWidth * peakWidth / 2 / (deltab - 0.0002) / (deltab - 0.0002)); //0.00029813225

			if (normi < peaks[0] - peakWidth){
				den = normi / (peaks[0] - peakWidth) * v1;
			}
			else if (normi < peaks[0] + peakWidth){
				int j = 0;
				float delta = deltab - 0.0002 * j, cooef = cooefb * (j + 1);
				den = cooef * exp(-(peaks[j] - normi)*(peaks[j] - normi) / 2 / delta / delta);
			}
			else if (normi > peaks[1] - peakWidth){
				int j = 1;
				float delta = deltab - 0.0002 * j, cooef = cooefb * (j + 1);
				den = cooef * exp(-(peaks[j] - normi)*(peaks[j] - normi) / 2 / delta / delta);
			}
			else{
				float r = (normi - (peaks[0] + peakWidth)) / ((peaks[1] - peakWidth) - (peaks[0] + peakWidth));
				den = v1 * (1 - r) + v2 * r;
			}


			colorTable[i] = make_float4(col, den);
		}
	}
	else{
		for (int i = 0; i < numColors; i++){
			float funcRes = (i*1.0 / (numColors - 1) - rcp->transFuncP2) / (rcp->transFuncP1 - rcp->transFuncP2);
			funcRes = clamp(funcRes, 0.0, 1.0);
			colorTable[i] = make_float4(funcRes, funcRes, funcRes, funcRes);
		}
	}

	if (rcp->d_transferFunc == 0)
	{
		cudaChannelFormatDesc channelFloat4 = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaMallocArray(&(rcp->d_transferFunc), &channelFloat4, numColors, 1));
	}
	checkCudaErrors(cudaMemcpyToArray((rcp->d_transferFunc), 0, 0, &(colorTable[0]), sizeof(float4)*numColors, cudaMemcpyHostToDevice));
}


void VolumeRenderableImmerCUDA::updateColorTable()
{
	if (colorTable.size() != numColors){
		colorTable.resize(numColors);
	}
	if (rcp->useColor){
		float peakRange = 0.01;
		std::vector<float> peaks;
		peaks.push_back(rcp->transFuncP2);
		peaks.push_back(rcp->transFuncP1);

		for (int i = 0; i < numColors; i++){
			float normi = i*1.0 / (numColors - 1);
			float colRes;

			if (peaks.size() < 2){
				colRes = normi;
			}
			else{
				colRes = clamp((normi - peaks[0]) / (peaks.back() - peaks[0]), 0.0, 1.0);
			}
			float3 col = make_float3(colRes, 0, 1 - colRes);
			float den = 0.00;

			/*
						float peakWidth = 0;
						float deltab = 0.012, cooefb = 80;
						if (peakWidth == 0){
						for (int j = 0; j < peaks.size(); j++){

						float delta = deltab - 0.002 * j, cooef = cooefb * (j + 1);

						float curDen = cooef * exp(-(peaks[j] - normi)*(peaks[j] - normi) / 2 / delta / delta);
						curDen = max(curDen - 0.1, 0);
						den = max(den, curDen);
						}
						}
						else{
						for (int j = 0; j < peaks.size(); j++){
						float delta = deltab / (j + 1), cooef = cooefb * (j + 1);

						float xv;
						if(abs(normi - peaks[j]) < peakWidth){
						xv = 0;
						}
						else{
						xv = fmin(abs(peaks[j] + peakWidth - normi), abs(peaks[j] - peakWidth - normi));
						}
						float curDen = cooef * exp(-xv*xv / 2 / delta / delta);
						den = max(den, curDen);
						}
						}*/


			float densityThr = 0.008;

			float slopeRange = 0.05;

			float right0 = 0.001, left1 = 0.0001;

			float peakWidth = 0.015;
			float delta[2] = { 0.005, 0.0045 }, cooef[2] = {10, 20};
			//exp(-0.01*0.01/2/0.0025/0.0025)*5 = 0.00167731313
			//exp(-0.01*0.01/2/0.002/0.002)*10 = 0.00003726653
			//float v1 = 0.00167731313, v2 = 0.00003726653;
			float v0 = cooef[0] * exp(-peakWidth * peakWidth / 2 / delta[0] / delta[0]);
			float v1 = cooef[1] * exp(-peakWidth * peakWidth / 2 / delta[1] / delta[1]);

			if (normi < peaks[0] - peakWidth - slopeRange){
				den = 0;
			}
			else if (normi < peaks[0] - peakWidth){
				float startPos = peaks[0] - peakWidth - slopeRange;
				float r = (normi - startPos) / ((peaks[0] - peakWidth) - startPos);
				den = densityThr * (1-r) + v0 * r;
			}
			else if (normi < peaks[0] + peakWidth){
				int j = 0;
				den = cooef[j] * exp(-(peaks[j] - normi)*(peaks[j] - normi) / 2 / delta[j] / delta[j]);
			}
			else if (normi > peaks[1] - peakWidth){
				int j = 1;
				den = cooef[j] * exp(-(peaks[j] - normi)*(peaks[j] - normi) / 2 / delta[j] / delta[j]);
			}
			else{


				float r = (normi - (peaks[0] + peakWidth)) / ((peaks[1] - peakWidth) - (peaks[0] + peakWidth));
				den = right0 * (1 - r) + left1 * r;

				//if (normi < peaks[1] - peakWidth - slopeRange){
				//	den = 0;
				//}
				//else{// normi < peaks[1] - peakWidth
				//	float startPos = peaks[1] - peakWidth - slopeRange;
				//	float r = (normi - startPos) / ((peaks[1] - peakWidth) - startPos);
				//	den = v1 * r;
				//}
			}
			
			
			colorTable[i] = make_float4(col, den);
		}
	}
	else{
		for (int i = 0; i < numColors; i++){
			float funcRes = (i*1.0 / (numColors - 1) - rcp->transFuncP2) / (rcp->transFuncP1 - rcp->transFuncP2);
			funcRes = clamp(funcRes, 0.0, 1.0);
			colorTable[i] = make_float4(funcRes, funcRes, funcRes, funcRes);
		}
	}

	if (rcp->d_transferFunc == 0)
	{
		cudaChannelFormatDesc channelFloat4 = cudaCreateChannelDesc<float4>();
		checkCudaErrors(cudaMallocArray(&(rcp->d_transferFunc), &channelFloat4, numColors, 1));
	}
	checkCudaErrors(cudaMemcpyToArray((rcp->d_transferFunc), 0, 0, &(colorTable[0]), sizeof(float4)*numColors, cudaMemcpyHostToDevice));
}


void VolumeRenderableImmerCUDA::draw(float modelview[16], float projection[16])
{
	if (!visible)
		return;

	RecordMatrix(modelview, projection);

	int winWidth, winHeight;
	actor->GetWindowSize(winWidth, winHeight);

	if (blendPreviousImage){
		//for color
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		// copy from pbo to texture
		qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		qgl->glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, volumeTex);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, winWidth, winHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, winWidth, winHeight);
		checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_inputImageTex_resource, volumeTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_inputImageTex_resource));
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&c_inputImageColorArray, cuda_inputImageTex_resource, 0, 0));

		//for depth
		qgl->glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, textureDepth);
		glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, winWidth, winHeight);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, localDepthArray);
		glBindTexture(GL_TEXTURE_2D, 0);
		cudaExtent dataSize = make_cudaExtent(winWidth, winHeight, 1);
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(localDepthArray, dataSize.width*sizeof(float), dataSize.width, dataSize.height);
		copyParams.dstArray = c_inputImageDepthArray;
		copyParams.extent = dataSize;
		copyParams.kind = cudaMemcpyHostToDevice;
		checkCudaErrors(cudaMemcpy3D(&copyParams));
		setInputImageInfo(c_inputImageDepthArray, c_inputImageColorArray);
	}

	QMatrix4x4 q_modelview = QMatrix4x4(modelview).transposed();
	QMatrix4x4 q_invMV = q_modelview.inverted();
	QVector4D q_eye4 = q_invMV.map(QVector4D(0, 0, 0, 1));
	float3 eyeInLocal = make_float3(q_eye4[0], q_eye4[1], q_eye4[2]);

	QMatrix4x4 q_projection = QMatrix4x4(projection).transposed();
	QMatrix4x4 q_mvp = q_projection*q_modelview;
	QMatrix4x4 q_invMVP = q_mvp.inverted();

	q_invMV.copyDataTo(invMVMatrix);
	q_invMVP.copyDataTo(invMVPMatrix); //copyDataTo() automatically copy in row-major order
	q_mvp.copyDataTo(MVPMatrix);
	q_modelview.copyDataTo(MVMatrix);
	q_modelview.normalMatrix().copyDataTo(NMatrix);
	VolumeRender_setConstants(MVMatrix, MVPMatrix, invMVMatrix, invMVPMatrix, NMatrix, &(volume->spacing), rcp.get());
	if (!isFixed){
		recordFixInfo(q_mvp, q_modelview);
	}

	//prepare the storage for output
	uint *d_output;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
		cuda_pbo_resource));
	checkCudaErrors(cudaMemset(d_output, 0, winWidth*winHeight * 4));

	if (volume != 0){
		VolumeRender_computeGradient(&(volume->volumeCuda), &volumeCUDAGradient);
		VolumeRender_setGradient(&volumeCUDAGradient);
		VolumeRender_setVolume(&(volume->volumeCuda));
	}
	else {
		std::cout << "data not well set for volume renderable" << std::endl;
		exit(0);
	}

	//compute the dvr
	if (blendPreviousImage){
		//it is better that when blendPreviousImage, the label volume can still be involved as in VolumeRender_renderImmer


		if (useClipRendering){
			std::cout << "blendPreviousImage + useClipRendering not implemented!" << std::endl;
		}
		else{
			if (usePreInt){
				VolumeRender_renderImmer_withPreBlend(d_output, winWidth, winHeight, eyeInLocal, volume->size, rcp.get(), densityBonus, usePreInt, useSplineInterpolation);
			}
			else{
				VolumeRender_renderWithDepthInput(d_output, winWidth, winHeight, rcp->density, rcp->brightness, eyeInLocal, volume->size, rcp->maxSteps, rcp->tstep, rcp->useColor, densityBonus);
			}
		}

	}
	else{
		//VolumeRender_renderImmer(d_output, winWidth, winHeight, eyeInLocal, volume->size, rcp.get(), positionBasedDeformProcessor.get(), usePreInt, useSplineInterpolation, useClipRendering);
		VolumeRender_renderImmer(d_output, winWidth, winHeight, eyeInLocal, volume->size, rcp.get(),
			positionBasedDeformProcessor->getTunnelStart(), positionBasedDeformProcessor->getTunnelEnd(), positionBasedDeformProcessor->getRectVerticalDir(), positionBasedDeformProcessor->r, positionBasedDeformProcessor->getDeformationScale(), positionBasedDeformProcessor->getDeformationScaleVertical(), positionBasedDeformProcessor->isColoringDeformedPart,
			usePreInt, useSplineInterpolation, useClipRendering);
	}

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	if (blendPreviousImage){
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//since previous image has been blended, they should be removed
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_inputImageTex_resource, 0));
	}

	// draw image from PBO
	glDisable(GL_DEPTH_TEST);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	// draw using texture
	// copy from pbo to texture
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	qgl->glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, volumeTex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, winWidth, winHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);


	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// draw textured quad

	auto functions12 = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_1_2>();
	functions12->glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0);
	glVertex2f(-1, -1);
	glTexCoord2f(1, 0);
	glVertex2f(1, -1);
	glTexCoord2f(1, 1);
	glVertex2f(1, 1);
	glTexCoord2f(0, 1);
	glVertex2f(-1, 1);
	glEnd();

	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	glEnable(GL_DEPTH_TEST);
}



void VolumeRenderableImmerCUDA::initTextureAndCudaArrayOfScreen()
{
	int winWidth, winHeight;
	actor->GetWindowSize(winWidth, winHeight);

	qgl->glGenBuffers(1, &pbo);
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	qgl->glBufferData(GL_PIXEL_UNPACK_BUFFER, winWidth*winHeight*sizeof(GLubyte)* 4, 0, GL_STREAM_DRAW);
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	// create texture for display
	qgl->glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &volumeTex);
	glBindTexture(GL_TEXTURE_2D, volumeTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, winWidth, winHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	if (blendPreviousImage){
		//for depth
		glGenTextures(1, &textureDepth);
		qgl->glActiveTexture(GL_TEXTURE5);
		glBindTexture(GL_TEXTURE_2D, textureDepth);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, winWidth, winHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
		glBindTexture(GL_TEXTURE_2D, 0);

		localDepthArray = new float[winWidth*winHeight];
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		cudaExtent dataSize = make_cudaExtent(winWidth, winHeight, 1);
		checkCudaErrors(cudaMalloc3DArray(&c_inputImageDepthArray, &channelDesc, dataSize, 0));
	}
}

void VolumeRenderableImmerCUDA::deinitTextureAndCudaArrayOfScreen()
{
	//it seems that pbo has already been deleted in VolumeRenderableCUDA
	if (cuda_pbo_resource != 0){
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
		cuda_pbo_resource = 0;
	}
	if (pbo != 0){
		qgl->glDeleteBuffers(1, &pbo);
		pbo = 0;
	}
	if (volumeTex != 0){
		glDeleteTextures(1, &volumeTex);
		volumeTex = 0;
	}
	if (blendPreviousImage){
		//for color
		if (cuda_inputImageTex_resource != 0){
			checkCudaErrors(cudaGraphicsUnregisterResource(cuda_inputImageTex_resource));
			cuda_inputImageTex_resource = 0;
		}
		//for depth
		if (localDepthArray != 0)
			delete[] localDepthArray;
		if (textureDepth != 0)
			glDeleteTextures(1, &textureDepth);
		if (c_inputImageDepthArray != 0)
			cudaFreeArray(c_inputImageDepthArray);
	}
}

void VolumeRenderableImmerCUDA::resize(int width, int height)
{
	bool oriVisibility = visible;
	visible = false;
	deinitTextureAndCudaArrayOfScreen();
	initTextureAndCudaArrayOfScreen();
	visible = oriVisibility;
}


void VolumeRenderableImmerCUDA::preIntTableNeedUpdate()
{
	updateColorTable();
	updatePreIntTabelNew(rcp->d_transferFunc);
}


void VolumeRenderableImmerCUDA::startClipRendering(std::shared_ptr<Volume> channelVolume)
{
	//other places changed, but not changed this function yet!!!
	//useClipRendering = true;
	//InitChannelVolumeTex(channelVolume);
}
