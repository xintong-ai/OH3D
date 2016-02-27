#include "SphereRenderable.h"
#include "glwidget.h"
//TODO:
//The corrent performance bottle neck is the rendering but not the displacement
//a more efficient way to draw sphere 
//http://11235813tdd.blogspot.com/2013/04/raycasted-spheres-and-point-sprites-vs.html
//The sample code can be found from
//http://tubafun.bplaced.net/public/sphere_shader.zip
//

//removing the following lines will cause runtime error
#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
using namespace std;

#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include "ShaderProgram.h"
#include "GLSphere.h"
#include <helper_math.h>
#include <ColorGradient.h>


void SphereRenderable::LoadShaders(ShaderProgram*& shaderProg)
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
	layout(location = 0) out vec4 FragColor;
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

void SphereRenderable::init()
{
	LoadShaders(glProg);
	//m_vao = std::make_unique<QOpenGLVertexArrayObject>();
	//m_vao->create();

	glyphMesh = std::make_unique<GLSphere>(1, 8);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	GenVertexBuffer(glyphMesh->GetNumVerts(),
		glyphMesh->GetVerts());

	initForPicking(glyphMesh->GetNumVerts(), glyphMesh->GetVerts());
}


void SphereRenderable::UpdateData()
{
}

void SphereRenderable::DrawWithoutProgram(float modelview[16], float projection[16], ShaderProgram* sp)
{
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexPosition"));

	for (int i = 0; i < pos.size(); i++) {
		glPushMatrix();

		float4 shift = pos[i];
		//float scale = pow(sphereSize[i], 0.333) * 0.01;

		//std::cout << sphereSize[i] << " ";

		//m_vao->bind();

		QMatrix4x4 q_modelview = QMatrix4x4(modelview);
		q_modelview = q_modelview.transposed();
		float3 cen = actor->DataCenter();
		qgl->glUniform4f(glProg->uniform("LightPosition"), 0, 0, std::max(std::max(cen.x, cen.y), cen.z) * 2, 1);

		if (snappedGlyphId != i){
			qgl->glUniform3fv(glProg->uniform("Ka"), 1, &sphereColor[i].x);
			qgl->glUniform1f(glProg->uniform("Scale"), glyphSizeScale[i] * (1 - glyphSizeAdjust) + glyphSizeAdjust);// 1);///*sphereSize[i] * */glyphSizeScale[i]);
		}
		else{
			qgl->glUniform3f(glProg->uniform("Ka"), 0.95f, 0.95f, 0.95f);
			qgl->glUniform1f(glProg->uniform("Scale"), glyphSizeScale[i] * (1 - glyphSizeAdjust) + glyphSizeAdjust*2);// 1);///*sphereSize[i] * */glyphSizeScale[i]);
		}

		qgl->glUniform3f(glProg->uniform("Kd"), 0.3f, 0.3f, 0.3f);
		qgl->glUniform3f(glProg->uniform("Ks"), 0.2f, 0.2f, 0.2f);
		qgl->glUniform1f(glProg->uniform("Shininess"), 5);
		qgl->glUniform3fv(glProg->uniform("Transform"), 1, &shift.x);
		
		qgl->glUniform1f(glProg->uniform("Bright"), glyphBright[i]);
		qgl->glUniformMatrix4fv(glProg->uniform("ModelViewMatrix"), 1, GL_FALSE, modelview);
		qgl->glUniformMatrix4fv(glProg->uniform("ProjectionMatrix"), 1, GL_FALSE, projection);
		qgl->glUniformMatrix3fv(glProg->uniform("NormalMatrix"), 1, GL_FALSE, q_modelview.normalMatrix().data());

		glDrawArrays(GL_QUADS, 0, glyphMesh->GetNumVerts());
		//glDrawElements(GL_TRIANGLES, glyphMesh->numElements, GL_UNSIGNED_INT, glyphMesh->indices);
		//m_vao->release();
		glPopMatrix();
	}

}

void SphereRenderable::draw(float modelview[16], float projection[16])
{
	if (!updated) {
		UpdateData();
		updated = true;
	}
	if (!visible)
		return;

	myRecordMatrix(modelview, projection);

	RecordMatrix(modelview, projection);
	ComputeDisplace();

	glProg->use();
	DrawWithoutProgram(modelview, projection, glProg);
	glProg->disable();
}


void SphereRenderable::GenVertexBuffer(int nv, float* vertex)
{
	//m_vao->bind();

	qgl->glGenBuffers(1, &vbo_vert);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float) * 3, vertex, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexPosition"));

	//m_vao->release();
}

SphereRenderable::SphereRenderable(std::vector<float4>& _spherePos, std::vector<float> _val)
//SphereRenderable::SphereRenderable(float4* _spherePos, int _sphereCnt, float* _val)
	:GlyphRenderable(_spherePos)
{ 
	val = _val; 
	sphereColor.assign(_spherePos.size(), make_float3(1.0f, 1.0f, 1.0f));
	ColorGradient cg;
	const float valMax = 350;
	const float valMin = 70;
	for (int i = 0; i < _spherePos.size(); i++) {
		float valScaled = (val[i] - valMin) / (valMax - valMin);
		cg.getColorAtValue(valScaled, sphereColor[i].x, sphereColor[i].y, sphereColor[i].z);
	}
}
