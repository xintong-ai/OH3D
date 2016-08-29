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
//using namespace std;

#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include "ShaderProgram.h"
#include "GLSphere.h"
#include <helper_math.h>
#include <ColorGradient.h>

//for linux
#include <float.h>

//void LoadPickingShaders(ShaderProgram*& shaderProg)


SphereRenderable::SphereRenderable(std::vector<float4> _spherePos, std::vector<float> _val)
//SphereRenderable::SphereRenderable(float4* _spherePos, int _sphereCnt, float* _val)
#ifdef USE_DEFORM
:DeformGlyphRenderable(_spherePos)
#else
: GlyphRenderable(_spherePos)
#endif
{
	val = _val;
	sphereColor.assign(_spherePos.size(), make_float3(1.0f, 1.0f, 1.0f));
	ColorGradient cg;
	float vMax = -FLT_MAX;
	float vMin = FLT_MAX;
	int n = pos.size();
	float v = 0;
	for (int i = 0; i < n; i++) {
		v = val[i];
		if (v > vMax)
			vMax = v;
		if (v < vMin)
			vMin = v;
	}
	vMax = vMax - 0.2 * (vMax - vMin);
	for (int i = 0; i < _spherePos.size(); i++) {
		float valScaled = (val[i] - vMin) / (vMax - vMin);
		cg.getColorAtValue(valScaled, sphereColor[i].x, sphereColor[i].y, sphereColor[i].z);
	}
}

void SphereRenderable::resetColorMap(COLOR_MAP cm)
{
	ColorGradient cg(cm);
	float vMax = -FLT_MAX;
	float vMin = FLT_MAX;
	int n = val.size();
	float v = 0;
	for (int i = 0; i < n; i++) {
		v = val[i];
		if (v > vMax)
			vMax = v;
		if (v < vMin)
			vMin = v;
	}
	for (int i = 0; i < val.size(); i++) {
		float valScaled = (val[i] - vMin) / (vMax - vMin);
		cg.getColorAtValue(valScaled, sphereColor[i].x, sphereColor[i].y, sphereColor[i].z);
	}
}

void SphereRenderable::init()
{
	GlyphRenderable::init();
	LoadShaders(glProg);
	//m_vao = std::make_shared<QOpenGLVertexArrayObject>();
	//m_vao->create();

	glyphMesh = std::make_shared<GLSphere>(1, 8);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	GenVertexBuffer(glyphMesh->GetNumVerts(),
		glyphMesh->GetVerts());

	initPickingDrawingObjects();
}

void SphereRenderable::LoadShaders(ShaderProgram*& shaderProg)
{
#define GLSL(shader) "#version 410\n" #shader
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



void SphereRenderable::GenVertexBuffer(int nv, float* vertex)
{
	//m_vao->bind();

	qgl->glGenBuffers(1, &vbo_vert);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float) * 3, vertex, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	//qgl->glEnableVertexAttribArray(glPickingProg->attribute("VertexPosition"));

	//m_vao->release();
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
			qgl->glUniform1f(glProg->uniform("Scale"), glyphSizeScale[i] * glyphSizeAdjust);// 1);///*sphereSize[i] * */glyphSizeScale[i]);
		}
		else{
			qgl->glUniform3f(glProg->uniform("Ka"), 0.95f, 0.95f, 0.95f);
			qgl->glUniform1f(glProg->uniform("Scale"), glyphSizeScale[i] * glyphSizeAdjust * 2);// 1);///*sphereSize[i] * */glyphSizeScale[i]);
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

	qgl->glDisableVertexAttribArray(glProg->attribute("VertexPosition"));
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void SphereRenderable::draw(float modelview[16], float projection[16])
{
	if (!updated) {
		UpdateData();
		updated = true;
	}


	RecordMatrix(modelview, projection);


	if (!visible)
		return;
#ifdef USE_DEFORM
	ComputeDisplace(modelview, projection);
#endif
	glProg->use();
	DrawWithoutProgram(modelview, projection, glProg);
	glProg->disable();
}


void SphereRenderable::UpdateData()
{
}

void SphereRenderable::initPickingDrawingObjects()
{

	//init shader
#define GLSL(shader) "#version 410\n" #shader
	//shader is from https://www.packtpub.com/books/content/basics-glsl-40-shaders
	//using two sides shading
	const char* vertexVS =
		GLSL(
		layout(location = 0) in vec3 VertexPosition;
	uniform mat4 ModelViewMatrix;
	uniform mat4 ProjectionMatrix;
	uniform vec3 Transform;
	uniform float Scale;
	void main()
	{
		mat4 MVP = ProjectionMatrix * ModelViewMatrix;
		gl_Position = MVP * vec4(VertexPosition * (Scale * 0.08) + Transform, 1.0);
	}
	);

	const char* vertexFS =
		GLSL(
		layout(location = 0) out vec4 FragColor;
	uniform float r;
	uniform float g;
	uniform float b;
	void main() {
		FragColor = vec4(r, g, b, 1.0);
	}
	);

	glPickingProg = new ShaderProgram;
	glPickingProg->initFromStrings(vertexVS, vertexFS);

	glPickingProg->addAttribute("VertexPosition");
	glPickingProg->addUniform("r");
	glPickingProg->addUniform("g");
	glPickingProg->addUniform("b");

	glPickingProg->addUniform("ModelViewMatrix");
	glPickingProg->addUniform("ProjectionMatrix");
	glPickingProg->addUniform("Transform");
	glPickingProg->addUniform("Scale");

	//init vertex buffer
	qgl->glGenBuffers(1, &vbo_vert_picking);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert_picking);
	qgl->glVertexAttribPointer(glPickingProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, glyphMesh->GetNumVerts() * sizeof(float) * 3, glyphMesh->GetVerts(), GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glPickingProg->attribute("VertexPosition"));
}


void SphereRenderable::drawPicking(float modelview[16], float projection[16], bool isForGlyph)
{
	RecordMatrix(modelview, projection);

	glPickingProg->use();

	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert_picking);
	qgl->glVertexAttribPointer(glPickingProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glEnableVertexAttribArray(glPickingProg->attribute("VertexPosition"));

	for (int i = 0; i < pos.size(); i++) {
		//glPushMatrix();

		int r, g, b;
		if (isForGlyph){
			r = ((i + 1) & 0x000000FF) >> 0;
			g = ((i + 1) & 0x0000FF00) >> 8;
			b = ((i + 1) & 0x00FF0000) >> 16;
		}
		else{
			char c = feature[i];
			r = ((c)& 0x000000FF) >> 0;
			g = ((c)& 0x0000FF00) >> 8;
			b = ((c)& 0x00FF0000) >> 16;
		}


		float4 shift = pos[i];
		QMatrix4x4 q_modelview = QMatrix4x4(modelview);
		q_modelview = q_modelview.transposed();

		qgl->glUniform1f(glPickingProg->uniform("Scale"), glyphSizeScale[i] * (1 - glyphSizeAdjust) + glyphSizeAdjust);
		qgl->glUniform3fv(glPickingProg->uniform("Transform"), 1, &shift.x);
		qgl->glUniformMatrix4fv(glPickingProg->uniform("ModelViewMatrix"), 1, GL_FALSE, modelview);
		qgl->glUniformMatrix4fv(glPickingProg->uniform("ProjectionMatrix"), 1, GL_FALSE, projection);
		qgl->glUniform1f(glPickingProg->uniform("r"), r / 255.0f);
		qgl->glUniform1f(glPickingProg->uniform("g"), g / 255.0f);
		qgl->glUniform1f(glPickingProg->uniform("b"), b / 255.0f);

		glDrawArrays(GL_QUADS, 0, glyphMesh->GetNumVerts());
		//glPopMatrix();
	}

	qgl->glDisableVertexAttribArray(glPickingProg->attribute("VertexPosition"));
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);

	glPickingProg->disable();
}