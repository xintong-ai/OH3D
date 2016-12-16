#ifndef VOLUMERENDERABLECUDAKERNEL_H
#define VOLUMERENDERABLECUDAKERNEL_H

#include <cuda_runtime.h>
#include "Volume.h"
#include <vector>

typedef unsigned int  uint;


extern "C" {
	void VolumeRender_init();
	void VolumeRender_deinit();

	void VolumeRender_render(uint *d_output, uint imageW, uint imageH,
		float density, float brightness,
		float3 eyeInWorld, int3 volumeSize, int maxSteps, float tstep, bool useColor);


	void VolumeRender_setVolume(const VolumeCUDA *volume);
	void VolumeRender_setGradient(const VolumeCUDA *volume);


	void VolumeRender_setConstants(float *MVMatrix, float *MVPMatrix, float *invMVMatrix, float *invMVPMatrix, float *NormalMatrix, float* _transFuncP1, float* _transFuncP2, float* _la, float* _ld, float* _ls, float3* _spacing);


	void VolumeRender_computeGradient(const VolumeCUDA *volumeCUDAInput, VolumeCUDA *volumeCUDAGradient);


	//the function is used to blend the object line lens into the DVR
	//lensPoints has 8 points, in the order of 4 center face vertices, then 4 front face vertices
	//note to turn off the regular drawing in LensRenderable to see the effect.
	void VolumeRender_render_withLensBlending(uint *d_output, uint imageW, uint imageH,
		float density, float brightness,
		float3 eyeInWorld, int3 volumeSize, int maxSteps, float tstep, bool useColor, std::vector<float3> lensPoints);
};

#endif //VOLUMERENDERABLECUDAKERNEL_H