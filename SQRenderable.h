#ifndef SQ_RENDERABLE_H
#define SQ_RENDERABLE_H

#include "GlyphRenderable.h"
#include <memory>

class ShaderProgram;
class QOpenGLVertexArrayObject;
class SQRenderable :public GlyphRenderable
{
	std::vector < float > val; // the 7 floating point number tensor values.
	std::vector<float4> verts;
	std::vector<int> nVerts;
	std::vector<float3> normals;
	std::vector<unsigned int> indices;
	std::vector<int> nIndices;
	void LoadShaders();

	unsigned int vbo_vert;
	unsigned int vbo_indices;
	unsigned int vbo_normals;
	std::unique_ptr<ShaderProgram> glProg;
	std::unique_ptr<QOpenGLVertexArrayObject> m_vao;

public:
	SQRenderable(std::vector<float4> _pos, std::vector < float > _val);
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void UpdateData() override;
};

#endif //SQ_RENDERABLE_H