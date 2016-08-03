#include <VRVolumeRenderableCUDA.h>
#include <VolumeRenderableCUDA.h>
#include <VRWidget.h>
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

void VRVolumeRenderableCUDA::draw(float modelview[16], float projection[16])
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
	drawWithGivenWinSize(modelview, projection, winWidth, winHeight);
}

void VRVolumeRenderableCUDA::drawWithGivenWinSize(float modelview[16], float projection[16], int winWidth, int winHeight)
{


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
	VolumeRender_setConstants(MVMatrix, MVPMatrix, invMVMatrix, invMVPMatrix, NMatrix, &isCutaway, &(volumeRenderable->transFuncP1), &(volumeRenderable->transFuncP2), &(volumeRenderable->la), &(volumeRenderable->ld), &(volumeRenderable->ls), &(volumeRenderable->volume->spacing));
	//if (!isFixed){
	//	recordFixInfo(q_mvp, q_modelview);
	//}

	//prepare the storage for output
	uint *d_output;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
		cuda_pbo_resource));
	checkCudaErrors(cudaMemset(d_output, 0, winWidth*winHeight * 4));

	////those should be well set already by volumeRenderable
	//ComputeDisplace(modelview, projection);
	//VolumeRender_setVolume(&(modelVolumeDeformer->volumeCUDADeformed));
	//VolumeRender_setGradient(&(modelVolumeDeformer->volumeCUDAGradient));

	//compute the dvr
	VolumeRender_render_test(d_output, winWidth, winHeight, volumeRenderable->density, volumeRenderable->brightness, eyeInWorld, volumeRenderable->volume->size, volumeRenderable->maxSteps, volumeRenderable->tstep, volumeRenderable->useColor);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	// display results
	glClear(GL_COLOR_BUFFER_BIT);

	// draw image from PBO
	glDisable(GL_DEPTH_TEST);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	// draw using texture
	// copy from pbo to texture
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	qgl->glActiveTexture(GL_TEXTURE1);
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
}

void VRVolumeRenderableCUDA::deinitTextureAndCudaArrayOfScreen()
{
	if (cuda_pbo_resource != 0)
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
	if (pbo != 0)
		qgl->glDeleteBuffers(1, &pbo);
	if (volumeTex != 0)
		glDeleteTextures(1, &volumeTex);
}

void VRVolumeRenderableCUDA::SetVRActor(VRWidget* _a) {
	vractor = _a;
}