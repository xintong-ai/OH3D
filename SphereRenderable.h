#ifndef SPHERE_RENDERABLE_H
#define SPHERE_RENDERABLE_H

#include "GlyphRenderable.h"
#include <QObject>
#include <memory>
class ShaderProgram;
class QOpenGLVertexArrayObject;
class GLSphere;

class SphereRenderable :public GlyphRenderable
{
public:
	void init() override;
	virtual void DrawWithoutProgram(float modelview[16], float projection[16], ShaderProgram* sp) override;
	void draw(float modelview[16], float projection[16]) override;
	void UpdateData() override;
	SphereRenderable(std::vector<float4>& _spherePos, std::vector<float> _val);

protected:
	void initPickingDrawingObjects();
	void drawPicking(float modelview[16], float projection[16], bool isForGlyph);

private:
	std::vector<float> val;// = nullptr;
	std::vector<float3> sphereColor;
	void GenVertexBuffer(int nv, float* vertex);
	virtual void LoadShaders(ShaderProgram*& shaderProg) override;
	unsigned int vbo_vert;
	std::unique_ptr<GLSphere> glyphMesh;
	//std::unique_ptr<QOpenGLVertexArrayObject> m_vao;
	bool updated = false;
};
#endif //SPHERE_RENDERABLE_H
