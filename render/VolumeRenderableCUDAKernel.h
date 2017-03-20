#ifndef VOLUMERENDERABLECUDAKERNEL_H
#define VOLUMERENDERABLECUDAKERNEL_H

#include <cuda_runtime.h>
#include "Volume.h"
#include "myDefineRayCasting.h"
#include <memory>
#include <vector>
#include "PositionBasedDeformProcessor.h"

typedef unsigned int  uint;

extern "C" {
	void VolumeRender_init();
	void VolumeRender_deinit();

	void VolumeRender_render(uint *d_output, uint imageW, uint imageH,
		float density, float brightness,
		float3 eyeInLocal, int3 volumeSize, int maxSteps, float tstep, bool useColor);

	void VolumeRender_renderWithDepthInput(uint *d_output, uint imageW, uint imageH, float density, float brightness, float3 eyeInLocal, int3 volumeSize, int maxSteps, float tstep, bool useColor, float densityBonus);


	void VolumeRender_renderImmer(uint *d_output, uint imageW, uint imageH,
		float3 eyeInLocal, int3 volumeSize, char* screenMark, RayCastingParameters* rcp);

	void VolumeRender_setVolume(const VolumeCUDA *volume);
	void VolumeRender_setGradient(const VolumeCUDA *volume);
	void VolumeRender_setLabelVolume(const VolumeCUDA *volume);


	void VolumeRender_setConstants(float *MVMatrix, float *MVPMatrix, float *invMVMatrix, float *invMVPMatrix, float *NormalMatrix, float* _transFuncP1, float* _transFuncP2, float* _la, float* _ld, float* _ls, float3* _spacing, RayCastingParameters* rcp);


	void VolumeRender_computeGradient(const VolumeCUDA *volumeCUDAInput, VolumeCUDA *volumeCUDAGradient);

	void LabelProcessor(uint imageW, uint imageH,
		float density, float brightness,
		float3 eyeInLocal, int3 volumeSize, int maxSteps, float tstep, bool useColor, char* screenMark, VolumeCUDA *volumeCUDALabel);

	void setInputImageInfo(const cudaArray_t c_inputImageDepthArray, const cudaArray_t c_inputImageColorArray);

};

#endif //VOLUMERENDERABLECUDAKERNEL_H