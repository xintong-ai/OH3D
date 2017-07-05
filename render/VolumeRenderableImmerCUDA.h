#ifndef VOLUMERENDERABLEIMMERSIVECUDA_H
#define VOLUMERENDERABLEIMMERSIVECUDA_H

#include "Volume.h"
#include "Renderable.h"
#include <memory>
#include <QObject>
#include <QOpenGLTexture>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>

class ScreenMarker;
class PositionBasedDeformProcessor;
struct RayCastingParameters;

//the difference from VolumeRenderableCUDA is with an extra label volume, and a screen marker
class VolumeRenderableImmerCUDA :public Renderable//, protected QOpenGLFunctions
{
	Q_OBJECT

	//the volume to render 
	std::shared_ptr<Volume> volume = 0;
	std::shared_ptr<PositionBasedDeformProcessor> positionBasedDeformProcessor;//may not be a good design
public:
	VolumeRenderableImmerCUDA(std::shared_ptr<Volume> _volume, std::shared_ptr<VolumeCUDA> _vl = 0, std::shared_ptr<PositionBasedDeformProcessor> p = 0);
	~VolumeRenderableImmerCUDA();

	void setVolume(std::shared_ptr<Volume> v, bool needMoreChange = false){
		volume = v;
		if (needMoreChange){
			//todo in the future;
		}
	};

	bool isFixed = false;

	std::shared_ptr<RayCastingParameters> rcp;

	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void resize(int width, int height)override;

	std::shared_ptr<Volume> getVolume(){
		return volume;
	}

	void setScreenMarker(std::shared_ptr<ScreenMarker> _sm){ sm = _sm; }
	void setBlending(bool b, float d = 1.0){ blendPreviousImage = b; densityBonus = d; };

private:
	VolumeCUDA volumeCUDAGradient;
	std::shared_ptr<ScreenMarker> sm;

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

	uint *d_output;

	//attributes used to blend the current image in pbo into the volume rendering
	bool blendPreviousImage = false;
	float densityBonus;

	//for belending
	//for color
	struct cudaGraphicsResource *cuda_inputImageTex_resource = 0;
	//for depth
	//to transfer the depth of the current depth buffer to cuda, three possible methods are:
	//testmethod == 1: store the depth in a opengl texture, then use a cuda_fiberImageDepthTex_resource to map it to CUDA array. Currently this mapping does not accept GL_DEPTH_COMPONENT, thus the depth value need to be first copied out into a host array, then be copied into a opengl float texure.
	//another untested similar way is to encode the depth value into the color channel, which is non-intuitive and introduce extra error
	//testmethod == 2: copy out the depth value into a host array, then	store the depth in a 2D cuda array. does not work currently
	//anothe untested similar way is to use 1D cuda array
	//testmethod == 3: store the depth in a 3D cuda array, then use it to init a 3D cuda texture.
	//so here use method 3
	unsigned int textureDepth = 0;
	cudaArray_t c_inputImageDepthArray = 0;
	cudaArray_t c_inputImageColorArray; //no allocation or deallocation
	float *localDepthArray = 0; //used to transfer opengl depth to cuda array
};

#endif