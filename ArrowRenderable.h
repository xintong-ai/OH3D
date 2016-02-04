#ifndef ARROW_RENDERABLE_H
#define ARROW_RENDERABLE_H

#include "GlyphRenderable.h"
//#include <memory>

class ShaderProgram;
class QOpenGLVertexArrayObject;
class GLArrow;

class ArrowRenderable :public GlyphRenderable
{
	float val; //used for coloring particles
	std::vector<float4> verts;
	std::vector<float3> normals;
	std::vector<unsigned int> indices;
	//std::vector<QMatrix4x4> rotations;
	void LoadShaders();

	unsigned int vbo_vert;
	unsigned int vbo_indices;
	unsigned int vbo_normals;
	std::unique_ptr<ShaderProgram> glProg;
	std::unique_ptr<QOpenGLVertexArrayObject> m_vao;
	std::unique_ptr<GLArrow> glyphMesh;
	std::vector<QMatrix4x4> rotations;

public:
	ArrowRenderable(std::vector<float4> _pos, std::vector<float3> _vec, std::vector < float > _val);
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void UpdateData() override;
};

#endif //ARROW_RENDERABLE_H