#ifndef VOLUMERENDERABLECUDA_H
#define VOLUMERENDERABLECUDA_H

#include "Volume.h"
#include "Renderable.h"
#include <memory>
#include <QObject>
#include <QOpenGLTexture>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
class ShaderProgram;
class QOpenGLVertexArrayObject;
class Fiber;
class Lens;

enum DEFORM_METHOD{
	PRINCIPLE_DIRECTION,
	DISTANCE_MAP,
	PROJECTIVE_DYNAMIC
};
enum VIS_METHOD{
	CUTAWAY,
	DEFORM
};


class VolumeRenderableCUDA :public Renderable//, protected QOpenGLFunctions
{
	Q_OBJECT
	
	//the volume to render 
	std::shared_ptr<Volume> volume = 0;

	VIS_METHOD vis_method = VIS_METHOD::DEFORM;
	DEFORM_METHOD deformMethod = DEFORM_METHOD::PROJECTIVE_DYNAMIC;
	
public:
	VolumeRenderableCUDA(std::shared_ptr<Volume> _volume);
	~VolumeRenderableCUDA();

	//cutaway or deform paramteres
	bool isFixed = false;
	float wallRotateTan = 0;
	int curDeformDegree = 1; //for deform by PRINCIPLE_DIRECTION & DISTANCE_MAP,
	int curAnimationDeformDegree = 0; //for deform by PROJECTIVE_DYNAMIC
	
	//NEK
	//lighting
	float la = 1.0, ld = 0.2, ls = 0.1;
	////MGHT2
	//transfer function
	float transFuncP1 = 0.55;
	float transFuncP2 = 0.13;
	float density = 1;
	//ray casting
	int maxSteps = 768;
	float tstep = 0.25f;
	float brightness = 1.0;


	////////MGHT2
	////lighting
	//float la = 1.0, ld = 0.2, ls = 0.7;
	////transfer function
	//float transFuncP1 = 0.44;// 0.55;
	//float transFuncP2 = 0.29;// 0.13;
	//float density = 1.25;
	////ray casting
	//int maxSteps = 512;
	//float tstep = 0.25f;
	//float brightness = 1.3;


	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void resize(int width, int height)override;

	bool useColor = false;

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
};

#endif