#ifdef WIN32
#include "windows.h"
#endif

#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>

#define qgl	QOpenGLContext::currentContext()->functions()

#include "shaderprogram.h"
#include "PolyRenderable.h"
#include <QMatrix4x4>
#include "MeshReader.h"

void PolyRenderable::init()
{
	loadShaders();
	m_vao = new QOpenGLVertexArrayObject();
	m_vao->create();

	//glEnable(GL_DEPTH_TEST);

	GenVertexBuffer(m->numElements,
		m->Faces_Triangles,
		m->Normals);
}

void PolyRenderable::loadShaders()
{

#define GLSL(shader) "#version 400\n" #shader
	//shader is from https://www.packtpub.com/books/content/basics-glsl-40-shaders
	//using two sides shading

	//the reason for flat shading is that the normal is not computed right
	//http://stackoverflow.com/questions/4703432/why-does-my-opengl-phong-shader-behave-like-a-flat-shader
	const char* vertexVS =
		GLSL(
		layout(location = 0) in vec3 VertexPosition;
	layout(location = 1) in vec3 VertexNormal;
	smooth out vec3 tnorm;
	out vec4 eyeCoords;

	uniform mat4 ModelViewMatrix;
	uniform mat3 NormalMatrix;
	uniform mat4 ProjectionMatrix;
	uniform vec3 Transform;
	void main()
	{
		mat4 MVP = ProjectionMatrix * ModelViewMatrix;

		tnorm = normalize(NormalMatrix * normalize(VertexNormal));

		eyeCoords = ModelViewMatrix *
			vec4(VertexPosition, 1.0);

		gl_Position = MVP * vec4(VertexPosition + Transform, 1.0);
	}
	);

	const char* vertexFS =
		GLSL(
		uniform vec4 LightPosition; // Light position in eye coords.
	uniform vec3 Ka; // Diffuse reflectivity
	uniform vec3 Kd; // Diffuse reflectivity
	uniform vec3 Ks; // Diffuse reflectivity
	uniform float Shininess;
	in vec4 eyeCoords;

	smooth in vec3 tnorm;

	layout(location = 0) out vec4 FragColor;

	vec3 phongModel(vec4 position, vec3 normal) {
		vec3 s = normalize(vec3(LightPosition - position));
		vec3 v = normalize(-position.xyz);
		vec3 r = reflect(-s, normal);
		vec3 ambient = Ka;
		float sDotN = max(dot(s, normal), 0.0);
		vec3 diffuse = Kd * sDotN;
		vec3 spec = vec3(0.0);
		if (sDotN > 0.0)
			spec = Ks *
			pow(max(dot(r, v), 0.0), Shininess);
		return ambient + diffuse + spec;
	}

	void main() {
		vec3 FrontColor;
		vec3 BackColor;
		//tnorm = normalize(tnorm);

		FrontColor = phongModel(eyeCoords, tnorm);

		BackColor = phongModel(eyeCoords, -tnorm);

		if (gl_FrontFacing) {
			FragColor = vec4(FrontColor, 1.0);//vec4(tnorm, 1.0);
		}
		else {
			FragColor = vec4(BackColor, 1.0);
		}
	}
	);

	glProg = new ShaderProgram();
	glProg->initFromStrings(vertexVS, vertexFS);

	glProg->addAttribute("VertexPosition");
	glProg->addAttribute("VertexNormal");

	glProg->addUniform("LightPosition");
	glProg->addUniform("Ka");
	glProg->addUniform("Kd");
	glProg->addUniform("Ks");
	glProg->addUniform("Shininess");

	glProg->addUniform("ModelViewMatrix");
	glProg->addUniform("NormalMatrix");
	glProg->addUniform("ProjectionMatrix");

	glProg->addUniform("Transform");
}

void PolyRenderable::resize(int width, int height)
{

}

void PolyRenderable::draw(float modelview[16], float projection[16])
{
	if (!visible)
		return;

	glMatrixMode(GL_MODELVIEW);


	glProg->use();
	m_vao->bind();

	qgl->glUniform4f(glProg->uniform("LightPosition"), 0, 0, 10, 1);
	qgl->glUniform3f(glProg->uniform("Ka"), ka.x, ka.y, ka.z);
	qgl->glUniform3f(glProg->uniform("Kd"), 0.6f, 0.6f, 0.6f);
	qgl->glUniform3f(glProg->uniform("Ks"), 0.2f, 0.2f, 0.2f);
	qgl->glUniform1f(glProg->uniform("Shininess"), 1);

	qgl->glUniform3fv(glProg->uniform("Transform"), 1, &transform.x);

	QMatrix4x4 q_modelview = QMatrix4x4(modelview);
	q_modelview = q_modelview.transposed();

	qgl->glUniformMatrix4fv(glProg->uniform("ModelViewMatrix"), 1, GL_FALSE, modelview);
	qgl->glUniformMatrix4fv(glProg->uniform("ProjectionMatrix"), 1, GL_FALSE, projection);
	qgl->glUniformMatrix3fv(glProg->uniform("NormalMatrix"), 1, GL_FALSE, q_modelview.normalMatrix().data());

	glDrawArrays(GL_TRIANGLES, 0, m->TotalConnectedTriangles * 3);
	//glDrawElements(GL_TRIANGLES, m->numElements, GL_UNSIGNED_INT, m->indices);

	//glBindVertexArray(0);
	m_vao->release();
	glProg->disable();

}
//
//void PolyRenderable::cleanup()
//{
//}

void PolyRenderable::GenVertexBuffer(int nv)
{
	m_vao->bind();

	qgl->glGenBuffers(1, &vbo_vert);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float) * 3, 0, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexPosition"));

	qgl->glGenBuffers(1, &vbo_norm);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_norm);
	qgl->glVertexAttribPointer(glProg->attribute("VertexNormal"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float) * 3, 0, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexNormal"));

	m_vao->release();
}

void PolyRenderable::GenVertexBuffer(int nv, float* vertex, float* normal)
{
	m_vao->bind();

	qgl->glGenBuffers(1, &vbo_vert);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float) * 3, vertex, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexPosition"));

	qgl->glGenBuffers(1, &vbo_norm);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_norm);
	qgl->glVertexAttribPointer(glProg->attribute("VertexNormal"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float) * 3, normal, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexNormal"));

	m_vao->release();
}