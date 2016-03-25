#ifndef POLY_RENDERABLE_H
#define POLY_RENDERABLE_H

#include "Renderable.h"

class ShaderProgram;
struct MeshReader;
class PolyRenderable :public Renderable
{
public:
	PolyRenderable(MeshReader* _m) { m = _m; }

	~PolyRenderable(){
		if (m != nullptr)
			delete m;
	}

    void GenVertexBuffer(int nv);

    void GenVertexBuffer(int nv, float* vertex, float* normal);

	void init() override;

    void resize(int width, int height) override;

    void draw(float modelview[16], float projection[16]) override;

    //void cleanup() override;

	float3 GetTransform(){
		return transform;
	}

	void SetTransform(float3 v){
		transform = v;
	}

	void SetAmbientColor(float r, float g, float b) {
		ka = make_float3(r, g, b);
	}

private:
	void loadShaders();

	MeshReader* m = nullptr;

protected:

    unsigned int vbo_norm;
	unsigned int vbo_vert;

	ShaderProgram *glProg;

	QOpenGLVertexArrayObject* m_vao;

	float3 transform = make_float3(0,0,0);

	float3 ka = make_float3(0.2f, 0.2f, 0.2f);
};
#endif //POLY_RENDERABLE_H
