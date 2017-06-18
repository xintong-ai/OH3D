#include "MatrixMgrRenderable.h"
#include "GLMatrixManager.h"
#include "TransformFunc.h"

#include "GLWidget.h"
#include "GLSphere.h"

//removing the following lines will cause runtime error
#ifdef WIN32
#include <windows.h>
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
//using namespace std;

#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include "ShaderProgram.h"

void MatrixMgrRenderable::init()
{
	m_vao = std::make_shared<QOpenGLVertexArrayObject>();
	m_vao->create();

	glyphMesh = std::make_shared<GLSphere>(1, 8);

	m_vao->bind();
	LoadShaders(glProg);

	GenVertexBuffer(glyphMesh->GetNumVerts(), glyphMesh->GetVerts());
}

void MatrixMgrRenderable::LoadShaders(ShaderProgram*& shaderProg)
{
#define GLSL(shader) "#version 440\n" #shader
	//shader is from https://www.packtpub.com/books/content/basics-glsl-40-shaders

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
		gl_Position = MVP * vec4(VertexPosition * (Scale * 0.08) + Transform, 1.0);
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
	//layout(location = 0) 
	out vec4 FragColor;
	uniform float Bright;


	vec3 phongModel(vec3 a, vec4 position, vec3 normal) {
		vec3 s = normalize(vec3(LightPosition - position));
		vec3 v = normalize(-position.xyz);
		vec3 r = reflect(-s, normal);
		vec3 ambient = a;// Ka * 0.8;
		float sDotN = max(dot(s, normal), 0.0);
		vec3 diffuse = Kd * sDotN;
		vec3 spec = vec3(0.0);
		if (sDotN > 0.0)
			spec = Ks *
			pow(max(dot(r, v), 0.0), Shininess);
		return ambient + diffuse + spec;
	}

	void main() {
		FragColor = vec4(Bright * phongModel(Ka * 0.5, eyeCoords, tnorm), 1.0);
		//FragColor = vec4(Bright * phongModel(Ka * 0.5, eyeCoords, tnorm), 0.5);
	}
	);

	shaderProg = new ShaderProgram;
	shaderProg->initFromStrings(vertexVS, vertexFS);

	shaderProg->addAttribute("VertexPosition");
	shaderProg->addUniform("LightPosition");
	shaderProg->addUniform("Ka");
	shaderProg->addUniform("Kd");
	shaderProg->addUniform("Ks");
	shaderProg->addUniform("Shininess");

	shaderProg->addUniform("ModelViewMatrix");
	shaderProg->addUniform("NormalMatrix");
	shaderProg->addUniform("ProjectionMatrix");

	shaderProg->addUniform("Transform");
	shaderProg->addUniform("Scale");
	shaderProg->addUniform("Bright");
}

void MatrixMgrRenderable::GenVertexBuffer(int nv, float* vertex)
{
	//m_vao->bind();

	qgl->glGenBuffers(1, &vbo_vert);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float)* 3, vertex, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	//qgl->glEnableVertexAttribArray(glPickingProg->attribute("VertexPosition"));

	//m_vao->release();
}

void MatrixMgrRenderable::draw(float modelview[16], float projection[16])
{
	if (!visible)
		return;

	RecordMatrix(modelview, projection);

	if (renderPart == 1){
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glLoadMatrixf(projection);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glLoadMatrixf(modelview);

		glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);
		glLineWidth(4);

		float _invmv[16];
		float _invpj[16];
		invertMatrix(projection, _invpj);
		invertMatrix(modelview, _invmv);

		int2 winSize = actor->GetWindowSize();
		float2 censcreen = make_float2(0.25*winSize.x, 0.25*winSize.y);
		float4 cen = Clip2ObjectGlobal(make_float4(make_float3(Screen2Clip(censcreen, winSize.x, winSize.y), 0.95), 1.0), _invmv, _invpj);

		float l = 2;
		glColor3f(0.90f, 0.10f, 0.10f);
		glBegin(GL_LINES);
		glVertex3f(cen.x, cen.y, cen.z);
		glVertex3f(cen.x + l, cen.y, cen.z);
		glEnd();

		glColor3f(0.90f, 0.90f, 0.10f);
		glBegin(GL_LINES);
		glVertex3f(cen.x, cen.y, cen.z);
		glVertex3f(cen.x, cen.y + l, cen.z);
		glEnd();

		glColor3f(0.10f, 0.90f, 0.10f);
		glBegin(GL_LINES);
		glVertex3f(cen.x, cen.y, cen.z);
		glVertex3f(cen.x, cen.y, cen.z + l);
		glEnd();


		glPopAttrib();
		//restore the original 3D coordinate system
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
	}
	else if (renderPart == 2){
		glProg->use();

		qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
		qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
		qgl->glEnableVertexAttribArray(glProg->attribute("VertexPosition"));
		m_vao->bind();

		glPushMatrix();

		float3 shift = matrixMgr->getEyeInLocal();

		QMatrix4x4 q_modelview = QMatrix4x4(modelview);
		q_modelview = q_modelview.transposed();
		float3 cen = actor->DataCenter();
		qgl->glUniform4f(glProg->uniform("LightPosition"), 0, 0, std::max(std::max(cen.x, cen.y), cen.z) * 2, 1);

		qgl->glUniform3f(glProg->uniform("Ka"), 0.0, 1.0, 0.0);
		qgl->glUniform1f(glProg->uniform("Scale"), 200);


		qgl->glUniform3f(glProg->uniform("Kd"), 0.3f, 0.3f, 0.3f);
		qgl->glUniform3f(glProg->uniform("Ks"), 0.2f, 0.2f, 0.2f);
		qgl->glUniform1f(glProg->uniform("Shininess"), 5);
		qgl->glUniform3fv(glProg->uniform("Transform"), 1, &shift.x);

		qgl->glUniform1f(glProg->uniform("Bright"), 1.0);
		qgl->glUniformMatrix4fv(glProg->uniform("ModelViewMatrix"), 1, GL_FALSE, modelview);
		qgl->glUniformMatrix4fv(glProg->uniform("ProjectionMatrix"), 1, GL_FALSE, projection);
		qgl->glUniformMatrix3fv(glProg->uniform("NormalMatrix"), 1, GL_FALSE, q_modelview.normalMatrix().data());

		glDrawArrays(GL_QUADS, 0, glyphMesh->GetNumVerts());
		//glDrawElements(GL_TRIANGLES, glyphMesh->numElements, GL_UNSIGNED_INT, glyphMesh->indices);
		//m_vao->release();
		glPopMatrix();

		m_vao->release();


		qgl->glDisableVertexAttribArray(glProg->attribute("VertexPosition"));
		qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);

		glProg->disable();
	}

}