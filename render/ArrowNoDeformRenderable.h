#ifndef ARROW_NODEFORM_RENDERABLE_H
#define ARROW_NODEFORM_RENDERABLE_H

#include <CMakeConfig.h>

#include "GlyphRenderable.h"

class ShaderProgram;
class QOpenGLVertexArrayObject;
class GLArrow;
class QOpenGLContext;


class ArrowNoDeformRenderable :public GlyphRenderable
{
	std::vector<float3> sphereColor;//used for coloring particles
	
	//regarding to the vectors and not included in Particle dataset
	std::vector<float3> vecs;
	float lMax, lMin; //max and min vector length
	std::vector<QMatrix4x4> rotations;


	unsigned int vbo_vert;
	unsigned int vbo_indices;
	unsigned int vbo_colors;
	unsigned int vbo_normals;
	std::shared_ptr<QOpenGLVertexArrayObject> m_vao;
	std::shared_ptr<GLArrow> glyphMesh;

	void GenVertexBuffer(int nv, float* vertex);


	bool initialized = false;

public:
	ArrowNoDeformRenderable(std::vector<float3> _vec, std::shared_ptr<Particle> _particle);
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void UpdateData() override;

	void setColorMap(COLOR_MAP cm);

protected:
	virtual void DrawWithoutProgram(float modelview[16], float projection[16], ShaderProgram* sp) override;
	virtual void LoadShaders(ShaderProgram*& shaderProg) override;

	void initPickingDrawingObjects();
	void drawPicking(float modelview[16], float projection[16], bool isForGlyph);
};

#endif //ARROW_RENDERABLE_H