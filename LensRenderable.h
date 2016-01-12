#ifndef LENS_RENDERABLE_H
#define LENS_RENDERABLE_H
#include "Renderable.h"

class Lens;
class LensRenderable :public Renderable
{
	std::vector<Lens*> lenses;
public:
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void UpdateData() override;
	LensRenderable(){}
	std::vector<Lens*> GetLenses() { return lenses; }
	//void AddSphereLens(int x, int y, int radius, float3 center);
	void AddCircleLens();

};
#endif