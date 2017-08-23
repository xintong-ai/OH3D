#ifndef POLY_RENDERABLE_H
#define POLY_RENDERABLE_H

#include "Renderable.h"
#include <memory>
class PositionBasedDeformProcessor;

class PolyMesh;
class ShaderProgram;
enum COLOR_MAP;

class PolyRenderable :public Renderable
{
public:
	PolyRenderable(std::shared_ptr<PolyMesh> p) { polyMesh = p; }

	~PolyRenderable(){
	}

	std::shared_ptr<PositionBasedDeformProcessor> positionBasedDeformProcessor = 0;//may not be a good design


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
	void setCenterBasedRendering(){ centerBasedRendering = true; }
	bool immersiveMode = false;

private:
	void loadShaders();
	void loadShadersImmer();

	std::shared_ptr<PolyMesh> polyMesh;

	bool centerBasedRendering = false;

protected:

    unsigned int vbo_norm;
	unsigned int vbo_vert;
	unsigned int vbo_val; //value for color

	ShaderProgram *glProg;
	ShaderProgram *glProgImmer; //specifically for the immersive deformation project. differences include color schemes for deformed part, etc.

	QOpenGLVertexArrayObject* m_vao;

	float3 transform = make_float3(0,0,0);

	float3 ka = make_float3(0.6f, 0.6f, 0.6f);


	void GenVertexBuffer(int nv, float* vertex, float* normal);
	void GenVertexBuffer(int nv);
};
#endif //POLY_RENDERABLE_H
