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
		gl_Position = MVP * vec4(VertexPosition * Scale + Transform, 1.0);
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

	void main() {
		vec3 unlitColor = 0.5 * vec3(1.0f, 1.0f, 1.0f);// GetColor2(norm, v);
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

	Renderable::draw(modelview, projection);
	glMatrixMode(GL_MODELVIEW);

	for (int i = 0; i < sphereCnt; i++) {
		glPushMatrix();

		float3 shift = spherePos[i];
		float scale = pow(sphereSize[i], 0.333) * 0.01;

		glProg->use();
		m_vao->bind();

		QMatrix4x4 q_modelview = QMatrix4x4(modelview);
		q_modelview = q_modelview.transposed();
		float3 cen = DataCenter();
		qgl->glUniform4f(glProg->uniform("LightPosition"), 0, 0, std::max(std::max(cen.x, cen.y), cen.z) * 2, 1);
		//qgl->glUniform3f(glProg->uniform("Ka"), 0.8f, 0.8f, 0.8f);
		qgl->glUniform3f(glProg->uniform("Kd"), 0.3f, 0.3f, 0.3f);
		qgl->glUniform3f(glProg->uniform("Ks"), 0.2f, 0.2f, 0.2f);
		qgl->glUniform1f(glProg->uniform("Shininess"), 5);
		qgl->glUniform3fv(glProg->uniform("Transform"), 1, &shift.x);
		qgl->glUniform1f(glProg->uniform("Scale"), scale);
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

float3 SphereRenderable::DataCenter()
{ 
	return (dataMin + dataMax) * 0.5; 
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
