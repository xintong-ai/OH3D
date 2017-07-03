#ifndef TRACE_RENDERABLE_H
#define TRACE_RENDERABLE_H

#include "Renderable.h"
#include <memory>
#include <vector>
class ShaderProgram;
class QOpenGLContext;
class StopWatchInterface;
//class QOpenGLVertexArrayObject;
class Particle;
enum COLOR_MAP;

class TraceRenderable : public Renderable
{
	Q_OBJECT

public:
	TraceRenderable(std::vector<std::shared_ptr<Particle>> _particle);
	~TraceRenderable();


	//virtual void setColorMap(COLOR_MAP cm, bool isReversed = false) {};
	bool colorByFeature = false;//when the particle has multi attributes or features, choose which attribute or color is used for color. currently a simple solution using bool

	void init() override;
	void draw(float modelview[16], float projection[16]) override;

	std::vector<std::shared_ptr<Particle>> particleSet;


private:
	std::vector<float3> sphereColor;

};
#endif