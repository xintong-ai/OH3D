#ifndef GRID_RENDERABLE_H
#define GRID_RENDERABLE_H
#include "Renderable.h"
#include <vector>
#include <vector_types.h>

//used to draw the images of the deformed grid, used in Xin's PacificVis streamline paper
//not designed in the same way with other renderables, and not mainteined currently

class GridRenderable : public Renderable
{
public:
	GridRenderable(int n);
	void UpdateGrid();

private:
	std::vector < float2 > grid;
	std::vector < float2 > mesh;
	int2 gridResolution = make_int2(8,8);
protected:
	void resize(int width, int height) override;
	void draw(float modelview[16], float projection[16]) override;
};

#endif //GRID_RENDERABLE_H