#ifndef MODEL_GRID_RENDERABLE_H
#define MODEL_GRID_RENDERABLE_H
#include "Renderable.h"
#include <memory>
class ModelGrid;

class ModelGridRenderable : public Renderable{
	ModelGrid* modelGrid;
	std::shared_ptr<float3> gridPts;
	unsigned int vertex_handle = 0;
	unsigned int triangle_handle = 0;
public:
	ModelGridRenderable(ModelGrid* _modelGrid);// float dmin[3], float dmax[3], int nPart);
protected:
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void mouseRelease(int x, int y, int modifier) override;

};
#endif