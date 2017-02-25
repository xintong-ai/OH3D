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
	VolumeRender_render(d_output, winWidth, winHeight, rcp->density, rcp->brightness, eyeInLocal, volume->size, rcp->maxSteps, rcp->tstep, rcp->useColor);


	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	// display results
	glClear(GL_COLOR_BUFFER_BIT);

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



	glProg->use();

	qgl->glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, volumeTex);
	qgl->glUniform1i(glProg->uniform("Tex_volume"), 0);
	//glBindTexture(GL_TEXTURE_2D, 0);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glProg->disable();

	/*
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
	*/

	glBindTexture(GL_TEXTURE_2D, 0);

	glEnable(GL_DEPTH_TEST);

}



void VolumeRenderableCUDAShader::initTextureAndCudaArrayOfScreen()
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
		void main()
		{
			vec2 quadVertices[4];
			quadVertices[0] = vec2(-1.0, -1.0);
			quadVertices[1] = vec2(1.0, -1.0);
			quadVertices[2] = vec2(-1.0, 1.0);
			quadVertices[3] = vec2(1.0, 1.0);

			gl_Position = vec4(quadVertices[gl_VertexID], -1.0, 1.0);
		}
	);

	const char* vertexFS =
		GLSL(
		layout(location = 0) out vec4 FragColor;
		out float gl_FragDepth;

		uniform sampler2D Tex_volume;

		void main() {
			//vec4 frag_in_clip = vec4((gl_FragCoord.x / winWidth - 0.5)*2.0, (gl_FragCoord.y / winHeight - 0.5)*2.0, -1.0, 1.0);
			vec2 samplePos = vec2(gl_FragCoord.x, gl_FragCoord.y);
			
			unsigned char c = texture(Tex_volume, samplePos)[0];
			vec4 res = texture(Tex_volume, samplePos)/16;
			
			//FragColor = vec4(gl_FragCoord.x / 100, gl_FragCoord.y /100, res.z, 1.0);
			FragColor = vec4(res.x, res.y, res.z, 1.0);

			//float xx = res.x;
			//FragColor = vec4(xx / 8, 0.0, 0.0, 1.0);

			//float xx = texture(Tex_volume, samplePos)[0];
			//FragColor = vec4(xx, 0.0, 0.0, 1.0);
			

			gl_FragDepth = 1.0;

		}
	);

	shaderProg = new ShaderProgram;
	shaderProg->initFromStrings(vertexVS, vertexFS);
	
	shaderProg->addUniform("Tex_volume");

	//qgl->glActiveTexture(GL_TEXTURE0);
	//glBindTexture(GL_TEXTURE_2D, volumeTex);
	//qgl->glUniform1i(glProg->uniform("Tex_volume"), 0);
	//glBindTexture(GL_TEXTURE_2D, 0);


}
