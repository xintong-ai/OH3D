#ifndef MATRIXMGR_RENDERABLE_H
#define MATRIXMGR_RENDERABLE_H
#include "Renderable.h"
#include <memory>

class GLMatrixManager;

class MatrixMgrRenderable :public Renderable
{
	Q_OBJECT
	
	std::shared_ptr<GLMatrixManager> matrixMgr;
public:

	void init() override
	{
	};
	void draw(float modelview[16], float projection[16]) override;
	MatrixMgrRenderable(std::shared_ptr<GLMatrixManager>  l){
		matrixMgr = l;
	};
	~MatrixMgrRenderable(){};

};
#endif