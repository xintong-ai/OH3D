#ifndef VOLUMERENDERABLECUDAKERNEL_H
#define VOLUMERENDERABLECUDAKERNEL_H

#include <cuda_runtime.h>
#include "Volume.h"


#define VOLUMERENDER_TFS              2
#define VOLUMERENDER_TF_PREINTSIZE    1024
#define VOLUMERENDER_TF_PREINTSTEPS   1024
#define VOLUMERENDER_TF_PREINTRAY     4

typedef unsigned int  uint;


extern "C" {
	void VolumeRender_init();
	void VolumeRender_deinit();

	void VolumeRender_render(uint *d_output, uint imageW, uint imageH,
		float density, float brightness,
		float3 eyeInWorld, int3 volumeSize, int maxSteps, float tstep, bool useColor);

	void VolumeRender_setVolume(const VolumeCUDA *volume);
	void VolumeRender_setGradient(const VolumeCUDA *volume);


	void VolumeRender_setConstants(float *MVMatrix, float *MVPMatrix, float *invMVMatrix, float *invMVPMatrix, float *NormalMatrix, bool *doCutaway, float* _transFuncP1, float* _transFuncP2, float* _la, float* _ld, float* _ls, float3* _spacing);


	void VolumeRender_computeGradient(const VolumeCUDA *volumeCUDAInput, VolumeCUDA *volumeCUDAGradient);
};

#endif //VOLUMERENDERABLECUDAKERNEL_H