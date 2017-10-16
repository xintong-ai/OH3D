#ifndef VOLUMERENDERABLECUDAKERNEL_H
#define VOLUMERENDERABLECUDAKERNEL_H

#include <cuda_runtime.h>
#include "Volume.h"
#include "myDefineRayCasting.h"
#include <memory>
#include <vector>

typedef unsigned int  uint;

extern "C" {
	void VolumeRender_init();
	void VolumeRender_deinit();
	
	void updatePreIntTabelNew(cudaArray *d_transferFunc);

	void VolumeRender_render(uint *d_output, uint imageW, uint imageH, float3 eyeInLocal, int3 volumeSize);
	void OmniVolumeRender_render(uint *d_output, uint imageW, uint imageH, float3 eyeInLocal, int3 volumeSize);

	void VolumeRender_renderWithDepthInput(uint *d_output, uint imageW, uint imageH, float density, float brightness, float3 eyeInLocal, int3 volumeSize, int maxSteps, float tstep, bool useColor, float densityBonus);


	void VolumeRender_renderImmer(uint *d_output, uint imageW, uint imageH,
		float3 eyeInLocal, int3 volumeSize, RayCastingParameters* rcp, 
		float3 tunnelStart, float3 tunnelEnd, float3 vertDir, float degree, float deformationscale, float deformationScaleVerticel, bool isColoringDeformedPart,
		bool usePreInt = false, bool useSplineInterpolation = false, bool useCliping = false);

	void VolumeRender_renderImmer_withPreBlend(uint *d_output, uint imageW, uint imageH,
		float3 eyeInLocal, int3 volumeSize, RayCastingParameters* rcp, float densityBonus,
		bool usePreInt = false, bool useSplineInterpolation = false);

	void VolumeRender_setVolume(const VolumeCUDA *volume);
	void VolumeRender_setGradient(const VolumeCUDA *volume);
	void VolumeRender_setLabelVolume(const VolumeCUDA *volume);


	void VolumeRender_setConstants(float *MVMatrix, float *MVPMatrix, float *invMVMatrix, float *invMVPMatrix, float *NormalMatrix, float3* _spacing, RayCastingParameters* rcp);


	void VolumeRender_computeGradient(const VolumeCUDA *volumeCUDAInput, VolumeCUDA *volumeCUDAGradient);

	void LabelProcessor(uint imageW, uint imageH,
		float density, float brightness,
		float3 eyeInLocal, int3 volumeSize, int maxSteps, float tstep, bool useColor, char* screenMark, VolumeCUDA *volumeCUDALabel);

	void setInputImageInfo(const cudaArray_t c_inputImageDepthArray, const cudaArray_t c_inputImageColorArray);

};

#endif //VOLUMERENDERABLECUDAKERNEL_H