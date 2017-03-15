#ifndef DEFORMFRAME_RENDERABLE_H
#define DEFORMFRAME_RENDERABLE_H
#include "Renderable.h"
#include <memory>
#include <vector>
#include <helper_math.h>

class GLMatrixManager;
class PositionBasedDeformProcessor;
class DeformFrameRenderable :public Renderable
{
	Q_OBJECT

	std::shared_ptr<GLMatrixManager> matrixMgr;
	std::shared_ptr<PositionBasedDeformProcessor> processor;

public:
	void init() override;
	void draw(float modelview[16], float projection[16]) override;

	DeformFrameRenderable(std::shared_ptr<GLMatrixManager> m, std::shared_ptr<PositionBasedDeformProcessor> p){
		matrixMgr = m;
		processor = p;
	};

	~DeformFrameRenderable()
	{
	};
};
#endif