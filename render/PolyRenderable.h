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

	bool useWireFrame = false;

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

	std::shared_ptr<PolyMesh> polyMesh;
	void dataChange();

private:
	void loadShaders();
	void loadShadersImmer();

	bool centerBasedRendering = false;


protected:

    unsigned int vbo_norm;
	unsigned int vbo_vert;
	unsigned int vbo_deviationVal; //value for coloring the deformed part of the data
	unsigned int vbo_val; //value for color

	ShaderProgram *glProg;
	ShaderProgram *glProgImmer; //specifically for the immersive deformation project. differences include color schemes for deformed part, etc.

	QOpenGLVertexArrayObject* m_vao;

	float3 transform = make_float3(0,0,0);

	float3 ka = make_float3(0.6f, 0.6f, 0.6f);
	float3 kd = make_float3(0.3f, 0.3f, 0.3f);
	float3 ks = make_float3(0.2f, 0.2f, 0.2f);

	void GenVertexBuffer(int nv, float* vertex, float* normal);
	//void GenVertexBuffer(int nv);
};
#endif //POLY_RENDERABLE_H
