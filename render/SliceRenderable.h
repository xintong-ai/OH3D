#ifndef SLICERENDERABLECUDA_H
#define SLICERENDERABLECUDA_H

#include "Volume.h"
#include "Renderable.h"
#include <memory>
#include <QObject>
#include <QOpenGLTexture>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>

struct RayCastingParameters;
class SliceRenderable :public Renderable//, protected QOpenGLFunctions
{
	Q_OBJECT

	//the volume to render 
	std::shared_ptr<Volume> volume = 0;

public:
	SliceRenderable(std::shared_ptr<Volume> _volume){ volume = _volume; };
	~SliceRenderable(){};


	std::shared_ptr<RayCastingParameters> rcp;

	void init() override;
	void draw(float modelview[16], float projection[16]) override;

	
private:
	
};

#endif