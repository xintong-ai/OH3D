//!!!NOTE!!!
//the current function uses glDrawPixels, which does not work well together with opengl viewport. as aa result, the image of the right eye currently does not render. should switch to use textures quads in the future

#ifndef VR_VOLUME_RENDERABLE_CUDA_H
#define VR_VOLUME_RENDERABLE_CUDA_H


#include "Renderable.h"
#include <memory>
class VolumeRenderableCUDA;
class VRWidget;
class Volume;
class RayCastingParameters;

class VRVolumeRenderableCUDA : public Renderable
{
	Q_OBJECT

	//different from other VR renderables, VRVolumeRenderableCUDA does not use a shared regular VolumeRenderableCUDA, because the heavily dependence on a different window size, and consequently, the size of the pre-allocated texture and/or pbo
	//VolumeRenderableCUDA* volumeRenderable;

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

	VRWidget* vractor;

	std::shared_ptr<Volume> volume = 0;

public:
	VRVolumeRenderableCUDA(std::shared_ptr<Volume> _v) {
		volume = _v;
	}

	void SetVRActor(VRWidget* _a) override;
	void resize(int width, int height)override;
	void init() override;
	void drawVR(float modelview[16], float projection[16], int eye) override;
	std::shared_ptr<RayCastingParameters> rcp;

};
#endif