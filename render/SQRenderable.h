#ifndef SQ_RENDERABLE_H
#define SQ_RENDERABLE_H

#include "GlyphRenderable.h"
#include <memory>

class TensorParticle;
class ShaderProgram;
class QOpenGLVertexArrayObject;
class SQRenderable :public GlyphRenderable
{
	//std::vector < float > val; // the 7 floating point number tensor values.
	std::vector<float4> verts;
	std::vector<int> nVerts;
	std::vector<float3> normals;
	std::vector<unsigned int> indices;
	std::vector<int> nIndices;
	std::vector<QMatrix4x4> rotations;

	unsigned int vbo_vert;
	unsigned int vbo_indices;
	unsigned int vbo_normals;

public:
	//SQRenderable(std::vector<float4> _pos, std::vector < float > _val);
	SQRenderable(std::shared_ptr<TensorParticle> p);
	void init() override;
	virtual void DrawWithoutProgram(float modelview[16], float projection[16], ShaderProgram* sp) override;
	void draw(float modelview[16], float projection[16]) override;
	void UpdateData();// override;


protected:
	virtual void LoadShaders(ShaderProgram*& shaderProg) override;

	void initPickingDrawingObjects();
	void drawPicking(float modelview[16], float projection[16], bool isForGlyph);
};

#endif //SQ_RENDERABLE_H