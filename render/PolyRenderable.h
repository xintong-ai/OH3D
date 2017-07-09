#ifndef POLY_RENDERABLE_H
#define POLY_RENDERABLE_H

#include "Renderable.h"
#include <memory>

class PolyMesh;
class ShaderProgram;
enum COLOR_MAP;

class PolyRenderable :public Renderable
{
public:
	PolyRenderable(std::shared_ptr<PolyMesh> p) { polyMesh = p; }

	~PolyRenderable(){
	}


    void GenVertexBuffer(int nv);

    void GenVertexBuffer(int nv, float* vertex, float* normal);

	void init() override;

    void resize(int width, int height) override;

    void draw(float modelview[16], float projection[16]) override;


	float3 GetTransform(){
		return transform;
	}

	void SetTransform(float3 v){
		transform = v;
	}

	void SetAmbientColor(float r, float g, float b) {
		ka = make_float3(r, g, b);
	}

	//float3 GetPolyCenter();
	bool isSnapped = false;

private:
	void loadShaders();
	std::shared_ptr<PolyMesh> polyMesh;

protected:

    unsigned int vbo_norm;
	unsigned int vbo_vert;

	ShaderProgram *glProg;

	QOpenGLVertexArrayObject* m_vao;

	float3 transform = make_float3(0,0,0);

	float3 ka = make_float3(0.2f, 0.2f, 0.2f);
};
#endif //POLY_RENDERABLE_H
