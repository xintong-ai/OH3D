#ifndef VOLUMERENDERABLECUDA_SHADER_H
#define VOLUMERENDERABLECUDA_SHADER_H

#include "Volume.h"
#include "Renderable.h"
#include <memory>
#include <QObject>
#include <QOpenGLTexture>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>

class ShaderProgram;

class VolumeRenderableCUDAShader :public Renderable//, protected QOpenGLFunctions
{
	Q_OBJECT

	//the volume to render 
	std::shared_ptr<Volume> volume = 0;

public:
	VolumeRenderableCUDAShader(std::shared_ptr<Volume> _volume);
	~VolumeRenderableCUDAShader();

	bool isFixed = false;

	std::shared_ptr<RayCastingParameters> rcp;

	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void resize(int width, int height)override;

	std::shared_ptr<Volume> getVolume(){
		return volume;
	}

private:
	VolumeCUDA volumeCUDAGradient;

	void initTextureAndCudaArrayOfScreen();
	void deinitTextureAndCudaArrayOfScreen();

	//texture and array for 2D screen
	GLuint pbo = 0;           // OpenGL pixel buffer object
	GLuint volumeTex = 0;     // OpenGL texture object
	struct cudaGraphicsResource *cuda_pbo_resource = 0; // CUDA Graphics Resource (to receive the result of the CUDA computer, and transfer it to PBO)

	GLuint pboDepth = 0;           // OpenGL pixel buffer object
	GLuint volumeTexDepth = 0;     // OpenGL texture object
	struct cudaGraphicsResource *cuda_pbo_resourceDepth = 0; 

	int2 lastPt = make_int2(0, 0);

	//to transfer data to cuda functions
	float MVMatrix[16];
	float MVPMatrix[16];
	float invMVMatrix[16];
	float invMVPMatrix[16];
	float NMatrix[9];

	QMatrix4x4 q_mvpFix, q_mvFix; //these matrices are used when the cutaway / deformation is not regarding to current camera position, but regarding to the camera position that caused the modified volume shape
	void recordFixInfo(QMatrix4x4 _q_mvp, QMatrix4x4 _q_mv)
	{
		q_mvpFix = _q_mvp;
		q_mvFix = _q_mv;

	};

	uint *d_output;

	ShaderProgram* glProg = nullptr;
	void LoadShaders(ShaderProgram*& shaderProg);

};

#endif