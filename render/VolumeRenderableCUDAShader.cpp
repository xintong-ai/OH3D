#include <time.h>

#include <vector>
/*
#include <QOpenGLVertexArrayObject>
#include <QOpenGLFunctions_1_2>
#include <QOpenGLFunctions_4_3_Core>
// removing the following lines will cause runtime error
#ifdef WIN32
#include <windows.h>
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
*/

#include <QOpenGLVertexArrayObject>
#include <QOpenGLFunctions_1_2>
#include <QOpenGLFunctions_4_3_Core>

// removing the following lines will cause runtime error
#ifdef WIN32
#include <windows.h>
#endif
#define qgl	QOpenGLContext::currentContext()->functions()

#include <QOpenGLFunctions>
#include "ShaderProgram.h"

#include <helper_math.h>
#include <cuda_gl_interop.h>

//#include "ShaderProgram.h"
#include "DeformGLWidget.h"
#include "VolumeRenderableCUDAShader.h"
#include "VolumeRenderableCUDAKernel.h"
#include "TransformFunc.h"


VolumeRenderableCUDAShader::VolumeRenderableCUDAShader(std::shared_ptr<Volume> _volume)
{
	volume = _volume;
	volumeCUDAGradient.VolumeCUDA_init(_volume->size, (float*)0, 1, 4);
}

VolumeRenderableCUDAShader::~VolumeRenderableCUDAShader()
{
	VolumeRender_deinit();
	deinitTextureAndCudaArrayOfScreen();
	//cudaDeviceReset();
};



void VolumeRenderableCUDAShader::init()
{
	VolumeRender_init();
	initTextureAndCudaArrayOfScreen();

	LoadShaders(glProg); //must be called after initTextureAndCudaArrayOfScreen()
}


void VolumeRenderableCUDAShader::draw(float modelview[16], float projection[16])
{
	if (!visible)
		return;

	RecordMatrix(modelview, projection);

	int winWidth, winHeight;
	actor->GetWindowSize(winWidth, winHeight);

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
	VolumeRender_setConstants(MVMatrix, MVPMatrix, invMVMatrix, invMVPMatrix, NMatrix, &(rcp->transFuncP1), &(rcp->transFuncP2), &(rcp->la), &(rcp->ld), &(rcp->ls), &(volume->spacing));
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

	
	float *d_outputDepth;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resourceDepth, 0));
	//size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_outputDepth, &num_bytes, cuda_pbo_resourceDepth));
	checkCudaErrors(cudaMemset(d_outputDepth, 0, winWidth*winHeight*sizeof(float)));
	
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
	//VolumeRender_render(d_output, winWidth, winHeight, rcp->density, rcp->brightness, eyeInLocal, volume->size, rcp->maxSteps, rcp->tstep, rcp->useColor);
	VolumeRender_renderWithDepthOutput(d_output, d_outputDepth, winWidth, winHeight, rcp->density, rcp->brightness, eyeInLocal, volume->size, rcp->maxSteps, rcp->tstep, rcp->useColor);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resourceDepth, 0));


	// draw image from PBO
	//glDisable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);

	glProg->use();

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	// draw using texture
	// copy from pbo to texture
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	qgl->glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, volumeTex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, winWidth, winHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	qgl->glUniform1i(glProg->uniform("Tex_volume"), 1); //should immedietaly call this after the value of the texture is assigned

	
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboDepth);
	qgl->glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, volumeTexDepth);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, winWidth, winHeight, GL_RED, GL_FLOAT, 0);
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	qgl->glUniform1i(glProg->uniform("Tex_depth"), 0);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindTexture(GL_TEXTURE_2D, 0); 
	
	glProg->disable();

	glBindTexture(GL_TEXTURE_2D, 0); //must be called after the draw is finished, or there may be unexpected error (the last texture may be disturbed)

	//glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

}



void VolumeRenderableCUDAShader::initTextureAndCudaArrayOfScreen()
{
	int winWidth, winHeight;
	actor->GetWindowSize(winWidth, winHeight);


	qgl->glGenBuffers(1, &pboDepth);
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboDepth);
	qgl->glBufferData(GL_PIXEL_UNPACK_BUFFER, winWidth*winHeight*sizeof(GLfloat), 0, GL_STREAM_DRAW);
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resourceDepth, pboDepth, cudaGraphicsMapFlagsWriteDiscard));

	// create texture for display
	glGenTextures(1, &volumeTexDepth); 
	qgl->glActiveTexture(GL_TEXTURE0);	
	glBindTexture(GL_TEXTURE_2D, volumeTexDepth);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, winWidth, winHeight, 0, GL_RED, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);


	qgl->glGenBuffers(1, &pbo);
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	qgl->glBufferData(GL_PIXEL_UNPACK_BUFFER, winWidth*winHeight*sizeof(GLubyte)* 4, 0, GL_STREAM_DRAW);
	qgl->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	// create texture for display
	glGenTextures(1, &volumeTex);
	qgl->glActiveTexture(GL_TEXTURE1);	
	glBindTexture(GL_TEXTURE_2D, volumeTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, winWidth, winHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); 
	glBindTexture(GL_TEXTURE_2D, 0);


}

void VolumeRenderableCUDAShader::deinitTextureAndCudaArrayOfScreen()
{
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
	if (cuda_pbo_resourceDepth != 0){
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resourceDepth));
		cuda_pbo_resourceDepth = 0;
	}
	if (pboDepth != 0){
		qgl->glDeleteBuffers(1, &pboDepth);
		pboDepth = 0;
	}
	if (volumeTexDepth != 0){
		glDeleteTextures(1, &volumeTexDepth);
		volumeTexDepth = 0;
	}
}

void VolumeRenderableCUDAShader::resize(int width, int height)
{
	visible = false;
	deinitTextureAndCudaArrayOfScreen();
	initTextureAndCudaArrayOfScreen();

	if (glProg != nullptr){
		delete glProg;
	}
	LoadShaders(glProg); //texture is changed in initTextureAndCudaArrayOfScreen()

	visible = true;
}

void VolumeRenderableCUDAShader::LoadShaders(ShaderProgram*& shaderProg)
{
#define GLSL(shader) "#version 440\n" #shader
	//shader is from https://www.packtpub.com/books/content/basics-glsl-40-shaders

	const char* vertexVS =
		GLSL(
		out vec2 textureCoords;

		void main()
		{
			vec2 quadVertices[4];
			quadVertices[0] = vec2(-1.0, -1.0);
			quadVertices[1] = vec2(1.0, -1.0);
			quadVertices[2] = vec2(-1.0, 1.0);
			quadVertices[3] = vec2(1.0, 1.0);

			gl_Position = vec4(quadVertices[gl_VertexID], -1.0, 1.0);
			textureCoords = (quadVertices[gl_VertexID] + 1.0) / 2.0;
		}
	);

	const char* vertexFS =
		GLSL(
		in vec2 textureCoords;
		layout(location = 0) out vec4 FragColor;
		out float gl_FragDepth;

		uniform sampler2D Tex_volume;
		uniform sampler2D Tex_depth;

		void main() {
			//FragColor = texture(Tex_volume, textureCoords);
			FragColor = texture(Tex_volume, textureCoords);
			gl_FragDepth = texture(Tex_depth, textureCoords)[0];
		}
	);

	shaderProg = new ShaderProgram;
	shaderProg->initFromStrings(vertexVS, vertexFS);
	
	shaderProg->addUniform("Tex_depth");
	shaderProg->addUniform("Tex_volume");

	//qgl->glActiveTexture(GL_TEXTURE0);
	//glBindTexture(GL_TEXTURE_2D, volumeTexDepth);
	//qgl->glUniform1i(glProg->uniform("Tex_depth"), 0);
	//glBindTexture(GL_TEXTURE_2D, 0);

	//qgl->glActiveTexture(GL_TEXTURE1);
	//glBindTexture(GL_TEXTURE_2D, volumeTex);
	//qgl->glUniform1i(glProg->uniform("Tex_volume"), 1);
	//glBindTexture(GL_TEXTURE_2D, 0);
}
