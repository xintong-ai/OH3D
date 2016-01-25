#ifndef SPHERE_RENDERABLE_H
#define SPHERE_RENDERABLE_H

#include "GlyphRenderable.h"
#include <QObject>
#include <memory>
class ShaderProgram;
class QOpenGLVertexArrayObject;
class GLSphere;
class SphereRenderable :public GlyphRenderable//, public QObject
{
	//Q_OBJECT
public:
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void UpdateData() override;
	SphereRenderable(std::vector<float4>& _spherePos, std::vector<float> _val);

	//void SetVolumeDim(int x, int y, int z){ dataDim[0] = x; dataDim[1] = y; dataDim[2] = z; }
private:
	std::vector<float> val;// = nullptr;
	std::vector<float3> sphereColor;
	void GenVertexBuffer(int nv, float* vertex);
	void LoadShaders();
	unsigned int vbo_vert;
	std::unique_ptr<GLSphere> glyphMesh;
	std::unique_ptr<ShaderProgram> glProg;

	std::unique_ptr<QOpenGLVertexArrayObject> m_vao;
	bool updated = false;
	//int dataMin[3];

};
#endif //SPHERE_RENDERABLE_H
