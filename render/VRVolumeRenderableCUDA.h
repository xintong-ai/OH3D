//maybe this class can be combined with VRGlyphRenderable.h to form a uniform VRRenderable

#ifndef VR_VOLUME_RENDERABLE_CUDA_H
#define VR_VOLUME_RENDERABLE_CUDA_H


#include "Renderable.h"
#include <memory>
class VolumeRenderableCUDA;
class ShaderProgram;

class VRVolumeRenderableCUDA : public Renderable
{
	Q_OBJECT

	VolumeRenderableCUDA* volumeRenderable;
	//ShaderProgram* glProg;

protected:
	std::vector<float> glyphBright;
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	//void resize(int width, int height) override;

public:
	VRVolumeRenderableCUDA(VolumeRenderableCUDA* _volumeRenderable) { volumeRenderable = _volumeRenderable; }
};
#endif