#ifndef MODELVOLUMEDEFORMER_H
#define MODELVOLUMEDEFORMER_H


#include <Volume.h>
#include <ModelGrid.h>

class ModelVolumeDeformer
{


public:
	VolumeCUDA volumeCUDADeformed;
	VolumeCUDA volumeCUDAGradient;

	Volume *originalVolume;

	ModelGrid *modelGrid;
	ModelVolumeDeformer(){};
	~ModelVolumeDeformer(){
		volumeCUDADeformed.VolumeCUDA_deinit();
	};

	void Init(Volume *ori);
	void SetModelGrid(ModelGrid* _modelGrid){ modelGrid = _modelGrid; }

	void deformByModelGrid();
	void computeGradient();
};
#endif