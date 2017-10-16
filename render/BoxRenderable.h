#ifndef BOX_RENDERABLE_H
#define BOX_RENDERABLE_H

#include "Renderable.h"

//used to draw a bounding box
class BoxRenderable:public Renderable
{
public:
	BoxRenderable(int3 d);
	BoxRenderable(float x, float y, float z, float nx, float ny, float nz);
	BoxRenderable(float3 _pos, float3 _dim);

	//void init() override;
	//void resize(int width, int height) override;
	virtual void draw(float modelview[16], float projection[16]) override;

private:
	float3 pos;
	float3 dim;
};
#endif //BOX_RENDERABLE_H
