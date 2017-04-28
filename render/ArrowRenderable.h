#ifndef ARROW_RENDERABLE_H
#define ARROW_RENDERABLE_H

#include "GlyphRenderable.h"

class ShaderProgram;
class QOpenGLVertexArrayObject;
class GLArrow;


class ArrowRenderable :public GlyphRenderable
{
public:
	void init() override;
	virtual void DrawWithoutProgram(float modelview[16], float projection[16], ShaderProgram* sp) override; 
	void draw(float modelview[16], float projection[16]) override;
	ArrowRenderable(std::vector<float3> _vec, std::shared_ptr<Particle> _particle);



protected:
	void initPickingDrawingObjects();
	void drawPicking(float modelview[16], float projection[16], bool isForGlyph);

private:
	std::vector<float3> vecs;
	std::vector<float3> cols;//used for coloring particles


	float lMax, lMin;
	std::vector<QMatrix4x4> rotations;

	std::vector<float4> verts;
	std::vector<float3> normals;
	std::vector<unsigned int> indices;
	//std::vector<QMatrix4x4> rotations;

	unsigned int vbo_vert, vbo_indices, vbo_colors, vbo_normals;
	std::shared_ptr<QOpenGLVertexArrayObject> m_vao;
	std::shared_ptr<GLArrow> glyphMesh;
	virtual void LoadShaders(ShaderProgram*& shaderProg) override;

};

#endif //ARROW_RENDERABLE_H