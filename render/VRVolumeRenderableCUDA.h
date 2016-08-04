//maybe this class can be combined with VRGlyphRenderable.h to form a uniform VRRenderable

#ifndef VR_VOLUME_RENDERABLE_CUDA_H
#define VR_VOLUME_RENDERABLE_CUDA_H


#include "Renderable.h"
#include <memory>
class VolumeRenderableCUDA;
class ShaderProgram;
class VRWidget;
class VRVolumeRenderableCUDA : public Renderable
{
	Q_OBJECT

	VolumeRenderableCUDA* volumeRenderable;
	//ShaderProgram* glProg;

	//texture and array for 2D screen
	unsigned int  pbo = 0;           // OpenGL pixel buffer object
	unsigned int  volumeTex = 0;     // OpenGL texture object
	struct cudaGraphicsResource *cuda_pbo_resource = 0; // CUDA Graphics Resource (to receive the result of the CUDA computer, and transfer it to PBO)
	void initTextureAndCudaArrayOfScreen();
	void deinitTextureAndCudaArrayOfScreen();

	unsigned int* pixelColor = 0;
	uint *d_output = 0;

	float MVMatrix[16];
	float MVPMatrix[16];
	float invMVMatrix[16];
	float invMVPMatrix[16];
	float NMatrix[9];

protected:
	void init() override;
	void draw(float modelview[16], float projection[16]) override;

	VRWidget* vractor;

public:
	VRVolumeRenderableCUDA(VolumeRenderableCUDA* _volumeRenderable) { volumeRenderable = _volumeRenderable; }
	void SetVRActor(VRWidget* _a) override;
	void resize(int width, int height)override;

};
#endif