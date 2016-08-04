#ifndef SPHERE_RENDERABLE_H
#define SPHERE_RENDERABLE_H

#include <CMakeConfig.h>
#ifdef USE_DEFORM
#include "DeformGlyphRenderable.h"
#else
#include "GlyphRenderable.h"
#endif

#include <QObject>
#include <memory>

class ShaderProgram;
class QOpenGLVertexArrayObject;
class GLSphere;
enum COLOR_MAP;
#ifdef USE_DEFORM
class SphereRenderable :public DeformGlyphRenderable
#else
class SphereRenderable :public GlyphRenderable
#endif
{
public:
	void init() override;
	virtual void DrawWithoutProgram(float modelview[16], float projection[16], ShaderProgram* sp) override;
	void draw(float modelview[16], float projection[16]) override;
	void UpdateData() override;
	SphereRenderable(std::vector<float4> _spherePos, std::vector<float> _val);

	virtual void resetColorMap(COLOR_MAP cm) override;

protected:
	void initPickingDrawingObjects();
	void drawPicking(float modelview[16], float projection[16], bool isForGlyph);

private:
	std::vector<float> val;// = nullptr;
	std::vector<float3> sphereColor;
	void GenVertexBuffer(int nv, float* vertex);
	virtual void LoadShaders(ShaderProgram*& shaderProg) override;
	unsigned int vbo_vert;
	std::shared_ptr<GLSphere> glyphMesh;
    std::shared_ptr<QOpenGLVertexArrayObject> m_vao;
	bool updated = false;
};
#endif //SPHERE_RENDERABLE_H
