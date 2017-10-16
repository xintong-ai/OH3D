

#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#ifdef WIN32
#include <windows.h>
#endif
#define qgl	QOpenGLContext::currentContext()->functions()

//#include <helper_math.h>
#include <helper_math.h>
#include <cuda_gl_interop.h>

#include "ShaderProgram.h"
#include "PolyRenderable.h"
#include <QMatrix4x4>
#include "PolyMesh.h"
#include "Particle.h"
#include "PositionBasedDeformProcessor.h"



void PolyRenderable::init()
{
	loadShaders();
	loadShadersImmer();
	m_vao = new QOpenGLVertexArrayObject();
	m_vao->create();

	//glEnable(GL_DEPTH_TEST);

//	GenVertexBuffer(m->numElements, m->Faces_Triangles, m->Normals);
	GenVertexBuffer(polyMesh->vertexcount, polyMesh->vertexCoords, polyMesh->vertexNorms);
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

		eyeCoords = ModelViewMatrix * vec4(VertexPosition, 1.0);

		gl_Position = MVP * vec4(VertexPosition + Transform, 1.0);
	}
	);

	const char* vertexFS =
		GLSL(
		uniform vec4 LightPosition; // Light position in eye coords.
	uniform vec3 Ka;
	uniform vec3 Kd;
	uniform vec3 Ks;
	uniform float Shininess;
	uniform float Opacity;
	
	smooth in vec3 tnorm;
	in vec4 eyeCoords;

	//layout(location = 0) 
	out vec4 FragColor;

	vec3 phongModel(vec4 position, vec3 normal) {
		vec3 s = normalize(vec3(LightPosition - position));
		vec3 v = normalize(-position.xyz);
		vec3 r = reflect(-s, normal);
		vec3 ambient = Ka;
		float sDotN = max(dot(s, normal), 0.0);
		vec3 diffuse = Kd * sDotN ;
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
			FragColor = vec4(FrontColor, Opacity);//vec4(tnorm, 1.0);
		}
		else {
			FragColor = vec4(BackColor, Opacity);
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
	glProg->addUniform("Opacity");

	glProg->addUniform("ModelViewMatrix");
	glProg->addUniform("NormalMatrix");
	glProg->addUniform("ProjectionMatrix");
	
	glProg->addUniform("Transform");
}

void PolyRenderable::loadShadersImmer()
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

		layout(location = 2) in float VertexDeviateVal;
		layout(location = 3) in float VertexColorVal;

		smooth out vec3 tnorm;
		out vec4 eyeCoords;
		out float deviateVal;
		out float colorVal;

		uniform mat4 ModelViewMatrix;
		uniform mat3 NormalMatrix;
		uniform mat4 ProjectionMatrix;
		uniform vec3 Transform;
		void main()
		{
			mat4 MVP = ProjectionMatrix * ModelViewMatrix;

			tnorm = normalize(NormalMatrix * normalize(VertexNormal));

			eyeCoords = ModelViewMatrix * vec4(VertexPosition, 1.0);

			gl_Position = MVP * vec4(VertexPosition + Transform, 1.0);
			deviateVal = VertexDeviateVal;
			colorVal = VertexColorVal;
		}
	);

	const char* vertexFS =
		GLSL(
		uniform vec4 LightPosition; // Light position in eye coords.
	uniform vec3 Ka;
	uniform vec3 Kd;
	uniform vec3 Ks;
	uniform float Shininess;
	uniform float Opacity;
	
	smooth in vec3 tnorm;
	in vec4 eyeCoords;
	in float deviateVal;
	in float colorVal;

	//layout(location = 0) 
	out vec4 FragColor;

	vec3 phongModel(vec4 position, vec3 normal) {
		//vec3 errorColor = vec3(1.0f, 0.0f, 0.0f);
		vec3 errorColor = vec3(0.02f, 0.94f, 0.04f);
		vec3 s = normalize(vec3(LightPosition - position));
		vec3 v = normalize(-position.xyz);
		vec3 r = reflect(-s, normal);
		//vec3 ambient = Ka * (1 - deviateVal) + errorColor*deviateVal;
		vec3 ccc = Ka*0.02 + (vec3(0.705882, 0.0156863, 0.14902)*colorVal + vec3(0.231373, 0.298039, 0.752941)*(1 - colorVal))*0.98;
		//vec3 ccc = Ka*0.000002 + (vec3(0.14902, 0.14902, 0.14902)*colorVal + vec3(0.231373, 0.231373, 0.231373)*(1 - colorVal))*0.98;
		
		vec3 ambient = ccc * (1 - deviateVal) + errorColor*deviateVal;
		float sDotN = max(dot(s, normal), 0.0);
		vec3 diffuse = Kd * sDotN ;
		vec3 spec = vec3(0.0);
		if (sDotN > 0.0)
			spec = Ks *
			pow(max(dot(r, v), 0.0), Shininess);
		return ambient + diffuse + spec;
	}

	void main() {
		vec3 FrontColor = phongModel(eyeCoords, tnorm);
		vec3 BackColor = phongModel(eyeCoords, -tnorm);

		if (gl_FrontFacing) {
			FragColor = vec4(FrontColor, Opacity);
		}
		else {
			FragColor = vec4(BackColor, Opacity);
		}
		//FragColor = vec4(FrontColor, Opacity);
	}
	);

	glProgImmer = new ShaderProgram();
	glProgImmer->initFromStrings(vertexVS, vertexFS);

	glProgImmer->addAttribute("VertexPosition");
	glProgImmer->addAttribute("VertexNormal");
	glProgImmer->addAttribute("VertexDeviateVal");
	glProgImmer->addAttribute("VertexColorVal");

	glProgImmer->addUniform("LightPosition");
	glProgImmer->addUniform("Ka");
	glProgImmer->addUniform("Kd");
	glProgImmer->addUniform("Ks");
	glProgImmer->addUniform("Shininess");
	glProgImmer->addUniform("Opacity");

	glProgImmer->addUniform("ModelViewMatrix");
	glProgImmer->addUniform("NormalMatrix");
	glProgImmer->addUniform("ProjectionMatrix");
	
	glProgImmer->addUniform("Transform");
}


void PolyRenderable::resize(int width, int height)
{

}


void PolyRenderable::draw(float modelview[16], float projection[16])
{
	if (!visible)
		return;

	if (polyMesh->verticesJustChanged){
		polyMesh->verticesJustChanged = false;
		dataChange();
	}


	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glMatrixMode(GL_MODELVIEW);

	ShaderProgram *curGlProg;
	if (immersiveMode && !centerBasedRendering){
		curGlProg = glProgImmer;
	}
	else{
		curGlProg = glProg;
	}
	curGlProg->use();
	m_vao->bind();

	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	qgl->glVertexAttribPointer(curGlProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, polyMesh->vertexcount * sizeof(float)* 3, polyMesh->vertexCoords, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_norm);
	qgl->glVertexAttribPointer(curGlProg->attribute("VertexNormal"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, polyMesh->vertexcount  * sizeof(float)* 3, polyMesh->vertexNorms, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);

	qgl->glUniform4f(curGlProg->uniform("LightPosition"), 0, 0, 1000, 1);

	qgl->glUniform3f(curGlProg->uniform("Kd"), kd.x, kd.y, kd.z);
	qgl->glUniform3f(curGlProg->uniform("Ks"), ks.x, ks.y, ks.z);
	qgl->glUniform1f(curGlProg->uniform("Shininess"), 1);
	qgl->glUniform1f(curGlProg->uniform("Opacity"), polyMesh->opacity);

	QMatrix4x4 q_modelview = QMatrix4x4(modelview);
	q_modelview = q_modelview.transposed();

	qgl->glUniformMatrix4fv(curGlProg->uniform("ModelViewMatrix"), 1, GL_FALSE, modelview);
	qgl->glUniformMatrix4fv(curGlProg->uniform("ProjectionMatrix"), 1, GL_FALSE, projection);
	qgl->glUniformMatrix3fv(curGlProg->uniform("NormalMatrix"), 1, GL_FALSE, q_modelview.normalMatrix().data());


	if (useWireFrame){
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glLineWidth(4);
		qgl->glUniform3f(curGlProg->uniform("Kd"), 0, 0, 0);
		qgl->glUniform3f(curGlProg->uniform("Ks"), 0, 0, 0);
	}

	if (centerBasedRendering){
		ka = make_float3(0.2f, 0, 0);

		int nRegion = polyMesh->particle->numParticles;
		for (int i = 0; i < nRegion; i++){
			//std::cout << "region " << i << " transform " << polyMesh->particle->pos[i].x << " " << polyMesh->particle->pos[i].y << " " << polyMesh->particle->pos[i].z<< std::endl;
			transform = make_float3(polyMesh->particle->pos[i].x, polyMesh->particle->pos[i].y, polyMesh->particle->pos[i].z);
			qgl->glUniform3fv(curGlProg->uniform("Transform"), 1, &transform.x);

			if (positionBasedDeformProcessor != 0 && positionBasedDeformProcessor->isColoringDeformedPart){
				float dis = length(
					make_float3(polyMesh->particle->pos[i].x, polyMesh->particle->pos[i].y, polyMesh->particle->pos[i].z)
					- make_float3(polyMesh->particle->posOrig[i].x, polyMesh->particle->posOrig[i].y, polyMesh->particle->posOrig[i].z));
				float scale;
				if (positionBasedDeformProcessor->getShapeModel() == SHAPE_MODEL::CUBOID){
					scale = positionBasedDeformProcessor->getDeformationScale();
				}
				else {
					scale = positionBasedDeformProcessor->radius;
				}
				float ratio = dis / (scale / 2);//0.5 is selected parameter
				ka = make_float3(0.2f, 0, 0) * (1 - ratio) + make_float3(0.0f, 0.2f, 0.2f) * ratio;
			}

			int startface = polyMesh->particle->valTuple[i * polyMesh->particle->tupleCount];
			int endface = polyMesh->particle->valTuple[i * polyMesh->particle->tupleCount + 1];
			int countface = endface - startface + 1;

			qgl->glUniform3f(curGlProg->uniform("Ka"), ka.x, ka.y, ka.z);
			glDrawElements(GL_TRIANGLES, countface * 3, GL_UNSIGNED_INT, polyMesh->indices + startface * 3);
			//std::cout << "start " << start << " end " << end << std::endl;
		}
	}
	else{
		if (immersiveMode){
			qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_deviationVal);
			qgl->glVertexAttribPointer(curGlProg->attribute("VertexDeviateVal"), 1, GL_FLOAT, GL_FALSE, 0, NULL);
			qgl->glBufferData(GL_ARRAY_BUFFER, polyMesh->vertexcount  * sizeof(float)* 1, polyMesh->vertexDeviateVals, GL_STATIC_DRAW);
			qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);

			qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_val);
			qgl->glVertexAttribPointer(curGlProg->attribute("VertexColorVal"), 1, GL_FLOAT, GL_FALSE, 0, NULL);
			qgl->glBufferData(GL_ARRAY_BUFFER, polyMesh->vertexcount * sizeof(float)* 1, polyMesh->vertexColorVals, GL_STATIC_DRAW);
			qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

		qgl->glUniform3fv(curGlProg->uniform("Transform"), 1, &transform.x);

		if (isSnapped)
			qgl->glUniform3f(curGlProg->uniform("Ka"), ka.x + 0.2, ka.y + 0.2, ka.z + 0.2);
		else{
			qgl->glUniform3f(curGlProg->uniform("Ka"), ka.x, ka.y, ka.z);
		}
		glDrawElements(GL_TRIANGLES, polyMesh->facecount * 3, GL_UNSIGNED_INT, polyMesh->indices);
	}

	if (useWireFrame)
	{
		//restore
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glLineWidth(1);
	}

	//glBindVertexArray(0);
	m_vao->release();
	curGlProg->disable();

	glDisable(GL_BLEND);




	////for test
	//if (centerBasedRendering){
	//	glMatrixMode(GL_PROJECTION);
	//	glPushMatrix();
	//	glLoadIdentity();
	//	glLoadMatrixf(projection);
	//	glMatrixMode(GL_MODELVIEW);
	//	glPushMatrix();
	//	glLoadIdentity();
	//	glLoadMatrixf(modelview);


	//	glColor4f(0.89f, 0.29f, 0.26f, 0.8f);
	//	glLineWidth(4);
	//	{

	//		int nRegion = polyMesh->particle->numParticles;
	//		for (int i = 0; i < nRegion; i++){
	//			float3 p1 = make_float3(polyMesh->particle->pos[i]);
	//			float3 p2 = p1 + 6 * make_float3(polyMesh->particle->valTuple[7], polyMesh->particle->valTuple[8], polyMesh->particle->valTuple[9]);
	//			glBegin(GL_LINES);
	//			glVertex3fv(&(p1.x));
	//			glVertex3fv(&(p2.x));
	//			glEnd();
	//		}

	//	}
	//	glLineWidth(1); //restore
	//}

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

/*
void PolyRenderable::GenVertexBuffer(int nv)
{
	m_vao->bind();

	qgl->glGenBuffers(1, &vbo_vert);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	if (immersiveMode && !centerBasedRendering){
	}
	else{
		qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
		qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float)* 3, 0, GL_STATIC_DRAW);
	}
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexPosition"));

	qgl->glGenBuffers(1, &vbo_norm);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_norm);
	if (immersiveMode && !centerBasedRendering){
	}
	else{
		qgl->glVertexAttribPointer(glProg->attribute("VertexNormal"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
		qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float)* 3, 0, GL_STATIC_DRAW);
	}
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexNormal"));

	m_vao->release();
}
*/
void PolyRenderable::GenVertexBuffer(int nv, float* vertex, float* normal)
{
	m_vao->bind();

	qgl->glGenBuffers(1, &vbo_vert);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	if (immersiveMode && !centerBasedRendering){
	}
	else{
		qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
		qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float)* 3, vertex, GL_STATIC_DRAW);
	}
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexPosition"));

	qgl->glGenBuffers(1, &vbo_norm);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_norm);
	if (immersiveMode && !centerBasedRendering){
	}
	else{
		qgl->glVertexAttribPointer(glProg->attribute("VertexNormal"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
		qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float)* 3, normal, GL_STATIC_DRAW);
	}
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexNormal"));

	if (immersiveMode && !centerBasedRendering){
		qgl->glGenBuffers(1, &vbo_deviationVal);
		qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_deviationVal);
		qgl->glVertexAttribPointer(glProgImmer->attribute("VertexDeviateVal"), 1, GL_FLOAT, GL_FALSE, 0, NULL);
		qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float)* 1, 0, GL_STATIC_DRAW);
		qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
		qgl->glEnableVertexAttribArray(glProgImmer->attribute("VertexDeviateVal"));

		qgl->glGenBuffers(1, &vbo_val);
		qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_val);
		qgl->glVertexAttribPointer(glProgImmer->attribute("VertexColorVal"), 1, GL_FLOAT, GL_FALSE, 0, NULL);
		qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float)* 1, 0, GL_STATIC_DRAW);
		qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
		qgl->glEnableVertexAttribArray(glProgImmer->attribute("VertexColorVal"));
	}

	m_vao->release();
}

//float3 PolyRenderable::GetPolyCenter(){
//	//return m->center;
//}

void PolyRenderable::dataChange()
{
	int nv = polyMesh->vertexcount;
	float* vertex = polyMesh->vertexCoords;
	float* normal = polyMesh->vertexNorms;

	m_vao->bind();

	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	if (immersiveMode && !centerBasedRendering){
	}
	else{
		qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
		qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float)* 3, vertex, GL_STATIC_DRAW);
	}
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);

	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_norm);
	if (immersiveMode && !centerBasedRendering){
	}
	else{
		qgl->glVertexAttribPointer(glProg->attribute("VertexNormal"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
		qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float)* 3, normal, GL_STATIC_DRAW);
	}
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (immersiveMode && !centerBasedRendering){
		qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_deviationVal);
		qgl->glVertexAttribPointer(glProgImmer->attribute("VertexDeviateVal"), 1, GL_FLOAT, GL_FALSE, 0, NULL);
		qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float)* 1, 0, GL_STATIC_DRAW);
		qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);

		qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_val);
		qgl->glVertexAttribPointer(glProgImmer->attribute("VertexColorVal"), 1, GL_FLOAT, GL_FALSE, 0, NULL);
		qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float)* 1, 0, GL_STATIC_DRAW);
		qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	m_vao->release();

}
