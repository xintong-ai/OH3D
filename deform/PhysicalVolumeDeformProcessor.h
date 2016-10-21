#ifndef	PHYSICAL_VOLUME_DEFORM_PROCESSOR_H
#define PHYSICAL_VOLUME_DEFORM_PROCESSOR_H
#include "Volume.h"
#include "MeshDeformProcessor.h"

class VolumeCUDA;

class PhysicalVolumeDeformProcessor
{
public:
	VolumeCUDA volumeCUDADeformed;
	std::shared_ptr<Volume> originalVolume;
	std::shared_ptr<MeshDeformProcessor> meshDeformer;

	PhysicalVolumeDeformProcessor(std::shared_ptr<MeshDeformProcessor> _modelGrid, std::shared_ptr<Volume> ori)
	{
		meshDeformer = _modelGrid;
		InitFromVolume(ori);
	};	
	~PhysicalVolumeDeformProcessor(){
		volumeCUDADeformed.VolumeCUDA_deinit();
	};

	void deformByMesh();

private:
	void InitFromVolume(std::shared_ptr<Volume> ori);

};
#endif