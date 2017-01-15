#include "VRVolumeRenderableCUDA.h"
#include "VolumeRenderableCUDA.h"
#include "VRWidget.h"
#include "VolumeRenderableCUDAKernel.h"

#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>

#include <cuda_gl_interop.h>

#include <QOpenGLFunctions_1_2>
//removing the following lines will cause runtime error
#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
#include "ShaderProgram.h"

// Standard includes
#include <iostream>

void VRVolumeRenderableCUDA::init()
{
	//glProg = new ShaderProgram;
	//glyphRenderable->LoadShaders(glProg);

	initTextureAndCudaArrayOfScreen();
}

void VRVolumeRenderableCUDA::drawVR(float modelview[16], float projection[16], int eye)
{
	if (!visible)
		return;
	//RecordMatrix(modelview, projection);
	//glColor3d(1, 1, 1);
	//glMatrixMode(GL_MODELVIEW);
	////glProg->use();
	//glColor3d(1, 0.3, 1);
	////glyphRenderable->DrawWithoutProgram(modelview, projection, glProg);
	//volumeRenderable->draw(modelview, projection);
	//glProg->disable();

	
	int winWidth, winHeight;
	vractor->GetWindowSize(winWidth, winHeight);

	winWidth = winWidth / 2;


	//drawWithGivenWinSize(modelview, projection, winWidth, winHeight);
	QMatrix4x4 q_modelview = QMatrix4x4(modelview).transposed();
	QMatrix4x4 q_invMV = q_modelview.inverted();
	QVector4D q_eye4 = q_invMV.map(QVector4D(0, 0, 0, 1));
	float3 eyeInWorld = make_float3(q_eye4[0], q_eye4[1], q_eye4[2]);

	QMatrix4x4 q_projection = QMatrix4x4(projection).transposed();
	QMatrix4x4 q_mvp = q_projection*q_modelview;
	QMatrix4x4 q_invMVP = q_mvp.inverted();

	q_invMV.copyDataTo(invMVMatrix);
	q_invMVP.copyDataTo(invMVPMatrix); //copyDataTo() automatically copy in row-major order
	q_mvp.copyDataTo(MVPMatrix);
	q_modelview.copyDataTo(MVMatrix);
	q_modelview.normalMatrix().copyDataTo(NMatrix);
	bool isCutaway = false;
	VolumeRender_setConstants(MVMatrix, MVPMatrix, invMVMatrix, invMVPMatrix, NMatrix, &(volumeRenderable->transFuncP1), &(volumeRenderable->transFuncP2), &(volumeRenderable->la), &(volumeRenderable->ld), &(volumeRenderable->ls), &(volumeRenderable->getVolume()->spacing));


	////those should be well set already by volumeRenderable
	//ComputeDisplace(modelview, projection);
	//VolumeRender_setVolume(&(modelVolumeDeformer->volumeCUDADeformed));
	//VolumeRender_setGradient(&(modelVolumeDeformer->volumeCUDAGradient));

	//compute the dvr
	VolumeRender_render(d_output, winWidth, winHeight, volumeRenderable->density, volumeRenderable->brightness, eyeInWorld, volumeRenderable->getVolume()->size, volumeRenderable->maxSteps, volumeRenderable->tstep, volumeRenderable->useColor);

	cudaMemcpy(pixelColor, d_output, winWidth*winHeight * sizeof(uint), cudaMemcpyDeviceToHost);
	glDrawPixels(winWidth, winHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixelColor);

}


void VRVolumeRenderableCUDA::resize(int width, int height)
{
	visible = false;
	deinitTextureAndCudaArrayOfScreen();
	initTextureAndCudaArrayOfScreen();
	visible = true;
}


void VRVolumeRenderableCUDA::initTextureAndCudaArrayOfScreen()
{
	int winWidth, winHeight;
	vractor->GetWindowSize(winWidth, winHeight);

	qgl->glGenBuffers(1, &pbo);
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	qgl->glBufferData(GL_PIXEL_UNPACK_BUFFER, winWidth*winHeight*sizeof(GLubyte)* 4, 0, GL_STREAM_DRAW);
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	// create texture for display
	qgl->glActiveTexture(GL_TEXTURE1);
	glGenTextures(1, &volumeTex);
	glBindTexture(GL_TEXTURE_2D, volumeTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, winWidth, winHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);


	pixelColor = new unsigned int[winWidth*winHeight];
	checkCudaErrors(cudaMalloc((void**)&d_output, sizeof(uint)*winWidth*winHeight));
	checkCudaErrors(cudaMemset(d_output, 0, winWidth*winHeight * sizeof(uint)));

}

void VRVolumeRenderableCUDA::deinitTextureAndCudaArrayOfScreen()
{
	if (cuda_pbo_resource != 0)
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
	if (pbo != 0)
		qgl->glDeleteBuffers(1, &pbo);
	if (volumeTex != 0)
		glDeleteTextures(1, &volumeTex);
	if (pixelColor != 0)
		delete[] pixelColor;
	if (d_output != 0)
		cudaFree(d_output);
}

void VRVolumeRenderableCUDA::SetVRActor(VRWidget* _a) {
	vractor = _a;
}