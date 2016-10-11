#ifndef MODELVOLUMEDEFORMER_H
#define MODELVOLUMEDEFORMER_H


#include <Volume.h>
#include <LineSplitModelGrid.h>

class ModelVolumeDeformer
{


public:
	VolumeCUDA volumeCUDADeformed;

	Volume *originalVolume;

	std::shared_ptr<LineSplitModelGrid> modelGrid;
	ModelVolumeDeformer(){};
	~ModelVolumeDeformer(){
		volumeCUDADeformed.VolumeCUDA_deinit();
	};

	void Init(Volume *ori);
	void SetModelGrid(std::shared_ptr<LineSplitModelGrid> _modelGrid){ modelGrid = _modelGrid; }

	void deformByModelGrid(float3 lensSpaceOrigin, float3 majorAxis, float3 lensDir, int3 nSteps, float step);

	void deformByModelGrid();
};
#endif