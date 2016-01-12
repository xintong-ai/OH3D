#ifndef SPHERE_RENDERABLE_H
#define SPHERE_RENDERABLE_H

#include "GlyphRenderable.h"
#include <QObject>
class ShaderProgram;
class QOpenGLVertexArrayObject;
class GLSphere;
class SphereRenderable :public GlyphRenderable, public QObject
{
	Q_OBJECT
public:
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void UpdateData() override;
	SphereRenderable(float4* _spherePos, int _sphereCnt, float* _sphereSize);

	//void SetVolumeDim(int x, int y, int z){ dataDim[0] = x; dataDim[1] = y; dataDim[2] = z; }
private:
	float* sphereSize = nullptr;
	void GenVertexBuffer(int nv, float* vertex);
	void LoadShaders();
	unsigned int vbo_vert;
	GLSphere* glyphMesh;
	ShaderProgram *glProg;

	QOpenGLVertexArrayObject* m_vao;
	bool updated = false;
	//int dataMin[3];

};
#endif //SPHERE_RENDERABLE_H
