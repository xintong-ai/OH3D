#ifndef MATRIXMGR_RENDERABLE_H
#define MATRIXMGR_RENDERABLE_H
#include "Renderable.h"
#include <memory>

class GLMatrixManager;

class GLSphere;
class ShaderProgram; 
class QOpenGLVertexArrayObject;

class MatrixMgrRenderable :public Renderable
{
	Q_OBJECT

	std::shared_ptr<GLMatrixManager> matrixMgr;
public:
	int renderPart = 1; //1. draw coordinate. 2. draw center

	void init() override;
	void draw(float modelview[16], float projection[16]) override;

	MatrixMgrRenderable(std::shared_ptr<GLMatrixManager>  l){
		matrixMgr = l;
	};

	~MatrixMgrRenderable(){};

private:
	//currently when draw center, using sphere, and reuse the code in SphereRenderable
	ShaderProgram* glProg = nullptr;
	void GenVertexBuffer(int nv, float* vertex);
	void LoadShaders(ShaderProgram*& shaderProg);
	unsigned int vbo_vert;
	std::shared_ptr<GLSphere> glyphMesh;
	std::shared_ptr<QOpenGLVertexArrayObject> m_vao;
};
#endif