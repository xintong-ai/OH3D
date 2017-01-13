#ifndef INTERACTOR_H
#define INTERACTOR_H
#include <memory>

class GLMatrixManager;
class Interactor
{
public:
	Interactor(){};
	~Interactor(){};
	
	virtual void Rotate(float fromX, float fromY, float toX, float toY, std::shared_ptr<GLMatrixManager> ){ return; };

	virtual void Translate(float x, float y, std::shared_ptr<GLMatrixManager>){ return; };
	virtual void wheelEvent(float v, std::shared_ptr<GLMatrixManager>){ return; };

};
#endif