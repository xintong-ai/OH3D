#ifndef MODELVOLUMEDEFORMER_H
#define MODELVOLUMEDEFORMER_H


#include <Volume.h>
#include <MeshDeformProcessor.h>

class ModelVolumeDeformer
{


public:
	VolumeCUDA volumeCUDADeformed;

	Volume *originalVolume;

	std::shared_ptr<MeshDeformProcessor> modelGrid;
	ModelVolumeDeformer(){};
	~ModelVolumeDeformer(){
		volumeCUDADeformed.VolumeCUDA_deinit();
	};

	void Init(Volume *ori);
	void SetModelGrid(std::shared_ptr<MeshDeformProcessor> _modelGrid){ modelGrid = _modelGrid; }

	void deformByModelGrid(float3 lensSpaceOrigin, float3 majorAxis, float3 lensDir, int3 nSteps, float step);

	void deformByModelGrid();
};
#endif