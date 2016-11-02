#ifndef	PHYSICAL_PARTICLE_DEFORM_PROCESSOR_H
#define PHYSICAL_PARTICLE_DEFORM_PROCESSOR_H
#include <memory>
#include <vector>
#include "Processor.h"

class Lens;
class LineLens3D;
class MeshDeformProcessor;
class Particle;
class PhysicalParticleDeformProcessor :public Processor
{
public:
	std::vector<Lens*> *lenses = 0;

	std::shared_ptr<Particle> particle;
	std::shared_ptr<MeshDeformProcessor> meshDeformer;

	PhysicalParticleDeformProcessor(std::shared_ptr<MeshDeformProcessor> _modelGrid, std::shared_ptr<Particle> _p)
	{
		particle = _p;
		meshDeformer = _modelGrid;
	};
	~PhysicalParticleDeformProcessor(){
	};

	void UpdatePointCoordsAndBright_LineMeshLens_Thrust(std::shared_ptr<Particle> p, float* brightness, LineLens3D * l, bool isFreezingFeature, int snappedFeatureId);
	void UpdatePointCoordsAndBright_UniformMesh(std::shared_ptr<Particle> p, float* brightness, float* _mv);
	bool process(float* modelview, float* projection, int winWidth, int winHeight) override;

private:

};
#endif