#include "CosmoRenderable.h"
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



CosmoRenderable::CosmoRenderable(std::shared_ptr<Particle> _particle)
#ifdef USE_DEFORM
:DeformGlyphRenderable(_particle)
#else
: GlyphRenderable(particle)
#endif
{
	sphereColor.assign(particle->numParticles, make_float3(1.0f, 1.0f, 1.0f));
	float vMax = particle->valMax;
	float vMin = particle->valMin;
	setColorMap(COLOR_MAP::SIMPLE_BLUE_RED);
}

void CosmoRenderable::setColorMap(COLOR_MAP cm)
{
	ColorGradient cg(cm);
	if (colorByFeature){
		float vMax = particle->featureMax;
		float vMin = particle->featureMin;
		for (int i = 0; i < particle->feature.size(); i++) {
			float valScaled = (particle->feature[i] - vMin) / (vMax - vMin);
			cg.getColorAtValue(valScaled, sphereColor[i].x, sphereColor[i].y, sphereColor[i].z);
		}
	}
	else{
		float vMax = particle->valMax;
		float vMin = particle->valMin;
		for (int i = 0; i < particle->val.size(); i++) {
			float valScaled = (particle->val[i] - vMin) / (vMax - vMin);
			cg.getColorAtValue(valScaled, sphereColor[i].x, sphereColor[i].y, sphereColor[i].z);
		}
	}
}

void CosmoRenderable::init()
{
	GlyphRenderable::init();
#ifdef USE_DEFORM
	DeformGlyphRenderable::init();
#endif

	m_vao = std::make_shared<QOpenGLVertexArrayObject>();
	m_vao->create();



	glyphMesh = std::make_shared<GLSphere>(1, 8);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	m_vao->bind();
	LoadShaders(glProg);

	GenVertexBuffer(glyphMesh->GetNumVerts(),
		glyphMesh->GetVerts());



	initPickingDrawingObjects();
	m_vao->release();
}

void CosmoRenderable::LoadShaders(ShaderProgram*& shaderProg)
{
#define GLSL(shader) "#version 150\n" #shader
	//shader is from https://www.packtpub.com/books/content/basics-glsl-40-shaders
	//using two sides shading
	const char* vertexVS =
		GLSL(
		//layout(location = 0) 
		in vec3 VertexPosition;
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



void CosmoRenderable::GenVertexBuffer(int nv, float* vertex)
{
	// m_vao->bind();

	qgl->glGenBuffers(1, &vbo_vert);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float)* 3, vertex, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	//qgl->glEnableVertexAttribArray(glPickingProg->attribute("VertexPosition"));

	//   m_vao->release();
}

void CosmoRenderable::DrawWithoutProgramold(float modelview[16], float projection[16], ShaderProgram* sp)
{
	m_vao->bind();
	for (int i = 0; i < particle->numParticles; i++) {
		glPushMatrix();

		float4 shift = pos[i];
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

		glPopMatrix();
	}
	m_vao->release();


}

void CosmoRenderable::DrawWithoutProgram(float modelview[16], float projection[16], ShaderProgram* sp)
{
	int2 winSize = actor->GetWindowSize();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(projection);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(modelview);



	glEnable(GL_POINT_SMOOTH);

	glDepthFunc(GL_LEQUAL);
	glDisable(GL_BLEND);
	const int numDotLayers = 3; //should be larger than 1, or the variable curColor needs redefine
	const float baseDotSize = 1.0;
	for (int i = 0; i < particle->numParticles; i++) {
	//	for (int i = 0; i < 10; i++) {

		for (int j = 0; j < numDotLayers; j++){

			if (j < numDotLayers-1){
				glEnable(GL_BLEND);
				glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			}
			
			if (snappedGlyphId != i){

				float coef = (numDotLayers - 1.0 - j) / (numDotLayers - 1.0);

				float curColorx = sphereColor[i].x + (1.0 - sphereColor[i].x) *coef;
				float curColory = sphereColor[i].y + (1.0 - sphereColor[i].y) *coef;
				float curColorz = sphereColor[i].z + (1.0 - sphereColor[i].z) *coef;

				float alpha = 1.0f / (numDotLayers - j);

				glColor4f(curColorx, curColory, curColorz, alpha);
				//glColor3f(curColorx, curColory, curColorz);

			}
			else{
				glColor3f(0.95f, 0.95f, 0.95f);
			}

			float size = baseDotSize*pow(2, numDotLayers - j - 1);
			glPointSize(size);
			glBegin(GL_POINTS);
			glVertex3f(pos[i].x , pos[i].y , pos[i].z );
			glEnd();

			if (j < numDotLayers - 1){
				glDisable(GL_BLEND);
			}
		}

	}
	glDisable(GL_POINT_SMOOTH);

	glDepthFunc(GL_LESS);


	//restore the original 3D coordinate system
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void CosmoRenderable::draw(float modelview[16], float projection[16])
{
	if (!updated) {
		UpdateData();
		updated = true;
	}


	RecordMatrix(modelview, projection);

#ifdef USE_DEFORM
	ComputeDisplace(modelview, projection);
#endif

	if (!visible)
		return;

	//glProg->use();
	DrawWithoutProgram(modelview, projection, glProg);
	//glProg->disable();
}


void CosmoRenderable::UpdateData()
{
}

void CosmoRenderable::initPickingDrawingObjects()
{

	//init shader
#define GLSL(shader) "#version 150\n" #shader
	//shader is from https://www.packtpub.com/books/content/basics-glsl-40-shaders
	//using two sides shading
	const char* vertexVS =
		GLSL(
		//	layout(location = 0) 
		in vec3 VertexPosition;
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
		//	layout(location = 0) 
		out vec4 FragColor;
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
	qgl->glBufferData(GL_ARRAY_BUFFER, glyphMesh->GetNumVerts() * sizeof(float)* 3, glyphMesh->GetVerts(), GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glPickingProg->attribute("VertexPosition"));
}


void CosmoRenderable::drawPicking(float modelview[16], float projection[16], bool isForGlyph)
{
	RecordMatrix(modelview, projection);

	glPickingProg->use();

	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert_picking);
	qgl->glVertexAttribPointer(glPickingProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glEnableVertexAttribArray(glPickingProg->attribute("VertexPosition"));

	for (int i = 0; i < particle->numParticles; i++) {
		//glPushMatrix();

		int r, g, b;
		//if (isForGlyph){
		//	r = ((i + 1) & 0x000000FF) >> 0;
		//	g = ((i + 1) & 0x0000FF00) >> 8;
		//	b = ((i + 1) & 0x00FF0000) >> 16;
		//}
		//else{
		//	char c = feature[i];
		//	r = ((c)& 0x000000FF) >> 0;
		//	g = ((c)& 0x0000FF00) >> 8;
		//	b = ((c)& 0x00FF0000) >> 16;
		//}
		r = ((i + 1) & 0x000000FF) >> 0;
		g = ((i + 1) & 0x0000FF00) >> 8;
		b = ((i + 1) & 0x00FF0000) >> 16;


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
