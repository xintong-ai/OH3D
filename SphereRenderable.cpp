#include "SphereRenderable.h"
#include "glwidget.h"

//removing the following lines will cause runtime error
#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()

#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include "ShaderProgram.h"
#include "GLSphere.h"
#include <helper_math.h>

void SphereRenderable::LoadShaders()
{
#define GLSL(shader) "#version 440\n" #shader
	//shader is from https://www.packtpub.com/books/content/basics-glsl-40-shaders
	//using two sides shading
	const char* vertexVS =
		GLSL(
	layout(location = 0) in vec3 VertexPosition;
	//layout(location = 1) in vec3 VertexNormal;
	smooth out vec3 tnorm;
	out vec4 eyeCoords;

	uniform mat4 ModelViewMatrix;
	uniform mat3 NormalMatrix;
	uniform mat4 ProjectionMatrix;
	uniform vec3 Transform;
	uniform float Scale;
	void main()
	{
		mat4 MVP = ProjectionMatrix * ModelViewMatrix;
		eyeCoords = ModelViewMatrix *
			vec4(VertexPosition, 1.0);
		tnorm = normalize(NormalMatrix * VertexPosition);
		gl_Position = MVP * vec4(VertexPosition * (pow(Scale, 0.333) * 0.01) + Transform, 1.0);
	}
	);

	const char* vertexFS =
		GLSL(
		uniform vec4 LightPosition; // Light position in eye coords.
	//uniform vec3 Ka; // Diffuse reflectivity
	uniform vec3 Kd; // Diffuse reflectivity
	uniform vec3 Ks; // Diffuse reflectivity
	uniform float Shininess;
	in vec4 eyeCoords;
	smooth in vec3 tnorm;
	layout(location = 0) out vec4 FragColor;
	uniform float Scale;

	vec3 phongModel(vec3 a, vec4 position, vec3 normal) {
		vec3 s = normalize(vec3(LightPosition - position));
		vec3 v = normalize(-position.xyz);
		vec3 r = reflect(-s, normal);
		vec3 ambient = a;// Ka;
		float sDotN = max(dot(s, normal), 0.0);
		vec3 diffuse = Kd * sDotN;
		vec3 spec = vec3(0.0);
		if (sDotN > 0.0)
			spec = Ks *
			pow(max(dot(r, v), 0.0), Shininess);
		return ambient + diffuse + spec;
	}

	vec4 GetColor(float v, float vmin, float vmax)
	{
		vec4 c = vec4(0.0, 0.0, 0.0, 1.0);
		float dv;
		dv = vmax - vmin;
		vec4 cm[3];
		cm[0] = vec4(0, 0, 1, 1);
		cm[1] = vec4(1, 1, 1, 1);
		cm[2] = vec4(1, 0, 0, 1);

		float pdv = 0.5 * dv;

		int i = 0;
		for (i = 0; i < 2; i++) {
			if (v >= (vmin + i*pdv) && v < (vmin + (i + 1)*pdv)) {
				c.r = cm[i].r + (v - vmin - i*pdv) / pdv * (cm[i + 1].r - cm[i].r);
				c.g = cm[i].g + (v - vmin - i*pdv) / pdv * (cm[i + 1].g - cm[i].g);
				c.b = cm[i].b + (v - vmin - i*pdv) / pdv * (cm[i + 1].b - cm[i].b);

				break;
			}
		}
		if (v == vmax) {
			c.r = cm[2].r;
			c.g = cm[2].g;
			c.b = cm[2].b;
		}
		return(c);
	}

	void main() {
		vec3 unlitColor = 0.5 * vec3(GetColor(Scale, 50, 400));// 0.5 * vec3(1.0f, 1.0f, 1.0f);// GetColor2(norm, v);
		FragColor = vec4(phongModel(unlitColor, eyeCoords, tnorm), 1.0);
	}
	);

	glProg = new ShaderProgram();
	glProg->initFromStrings(vertexVS, vertexFS);

	glProg->addAttribute("VertexPosition");
	glProg->addUniform("LightPosition");
	//glProg->addUniform("Ka");
	glProg->addUniform("Kd");
	glProg->addUniform("Ks");
	glProg->addUniform("Shininess");

	glProg->addUniform("ModelViewMatrix");
	glProg->addUniform("NormalMatrix");
	glProg->addUniform("ProjectionMatrix");

	glProg->addUniform("Transform");
	glProg->addUniform("Scale");
}

void SphereRenderable::init()
{
	LoadShaders();
	m_vao = new QOpenGLVertexArrayObject();
	m_vao->create();

	glyphMesh = new GLSphere(1, 8);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	GenVertexBuffer(glyphMesh->GetNumVerts(),
		glyphMesh->GetVerts());
}


void SphereRenderable::UpdateData()
{
}

void SphereRenderable::draw(float modelview[16], float projection[16])
{
	if (!updated) {
		UpdateData();
		updated = true;
	}
	if (!visible)
		return;

	RecordMatrix(modelview, projection);
	ComputeDisplace();

	glMatrixMode(GL_MODELVIEW);

	for (int i = 0; i < num; i++) {
		glPushMatrix();

		float4 shift = pos[i];
		//float scale = pow(sphereSize[i], 0.333) * 0.01;

		//std::cout << sphereSize[i] << " ";

		glProg->use();
		m_vao->bind();

		QMatrix4x4 q_modelview = QMatrix4x4(modelview);
		q_modelview = q_modelview.transposed();
		float3 cen = actor->DataCenter();
		qgl->glUniform4f(glProg->uniform("LightPosition"), 0, 0, std::max(std::max(cen.x, cen.y), cen.z) * 2, 1);
		//qgl->glUniform3f(glProg->uniform("Ka"), 0.8f, 0.8f, 0.8f);
		qgl->glUniform3f(glProg->uniform("Kd"), 0.3f, 0.3f, 0.3f);
		qgl->glUniform3f(glProg->uniform("Ks"), 0.2f, 0.2f, 0.2f);
		qgl->glUniform1f(glProg->uniform("Shininess"), 5);
		qgl->glUniform3fv(glProg->uniform("Transform"), 1, &shift.x);
		qgl->glUniform1f(glProg->uniform("Scale"), sphereSize[i]);
		qgl->glUniformMatrix4fv(glProg->uniform("ModelViewMatrix"), 1, GL_FALSE, modelview);
		qgl->glUniformMatrix4fv(glProg->uniform("ProjectionMatrix"), 1, GL_FALSE, projection);
		qgl->glUniformMatrix3fv(glProg->uniform("NormalMatrix"), 1, GL_FALSE, q_modelview.normalMatrix().data());

		glDrawArrays(GL_QUADS, 0, glyphMesh->GetNumVerts());
		//glDrawElements(GL_TRIANGLES, glyphMesh->numElements, GL_UNSIGNED_INT, glyphMesh->indices);
		m_vao->release();
		glProg->disable();
		glPopMatrix();
	}
}


void SphereRenderable::GenVertexBuffer(int nv, float* vertex)
{
	m_vao->bind();

	qgl->glGenBuffers(1, &vbo_vert);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float) * 3, vertex, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexPosition"));

	m_vao->release();
}

SphereRenderable::SphereRenderable(float4* _spherePos, int _sphereCnt, float* _sphereSize)
	:GlyphRenderable(_spherePos, _sphereCnt)
{ 
	sphereSize = _sphereSize; 
}
