#ifndef	PHYSICAL_VOLUME_DEFORM_PROCESSOR_H
#define PHYSICAL_VOLUME_DEFORM_PROCESSOR_H
#include <memory>
#include <vector>
#include "Processor.h"

class Lens;
class LineLens3D;
class MeshDeformProcessor;
class Volume;

class PhysicalVolumeDeformProcessor :public Processor
{
public:
	std::shared_ptr<Volume> volume;
	std::shared_ptr<MeshDeformProcessor> meshDeformer;

	PhysicalVolumeDeformProcessor(std::shared_ptr<MeshDeformProcessor> _modelGrid, std::shared_ptr<Volume> ori)
	{
		meshDeformer = _modelGrid;
		InitFromVolume(ori);
	};	

	~PhysicalVolumeDeformProcessor(){
	};

	bool process(float* modelview, float* projection, int winWidth, int winHeight) override;

private:
	void InitFromVolume(std::shared_ptr<Volume> ori);

};
#endif