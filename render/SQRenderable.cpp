#include "SQRenderable.h"
#include "glwidget.h"

#include <teem/ten.h>
//http://www.sci.utah.edu/~gk/vissym04/index.html

//removing the following lines will cause runtime error
#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()

#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include "ShaderProgram.h"
#include "Particle.h"

#include <memory>

using namespace std;

void SQRenderable::LoadShaders(ShaderProgram*& shaderProg) 
{

#define GLSL(shader) "#version 440\n" #shader
	//shader is from https://www.packtpub.com/books/content/basics-glsl-40-shaders


	const char* vertexVS =
		GLSL(
		layout(location = 0) in vec4 VertexPosition;
		layout(location = 1) in vec3 VertexNormal;
		smooth out vec3 tnorm;
		out vec4 eyeCoords;

		uniform mat4 ModelViewMatrix;
		uniform mat3 NormalMatrix;
		uniform mat4 ProjectionMatrix;
		uniform mat4 SQRotMatrix;
		uniform vec3 Transform;
		uniform float Scale;

		vec4 DivZ(vec4 v){
			return vec4(v.x / v.w, v.y / v.w, v.z / v.w, 1.0f);
		}

		void main()
		{
			mat4 MVP = ProjectionMatrix * ModelViewMatrix;
			eyeCoords = ModelViewMatrix * VertexPosition;
			tnorm = normalize(NormalMatrix * /*vec3(VertexPosition) + 0.001 * */VertexNormal);
			gl_Position = MVP * vec4(vec3(DivZ(SQRotMatrix * VertexPosition)) * 1000 * Scale + Transform, 1.0);
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
		out vec4 FragColor; //layout(location = 0) out vec4 FragColor;
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
	shaderProg->addAttribute("VertexNormal");
	shaderProg->addUniform("LightPosition");
	shaderProg->addUniform("Ka");
	shaderProg->addUniform("Kd");
	shaderProg->addUniform("Ks");
	shaderProg->addUniform("Shininess");

	shaderProg->addUniform("ModelViewMatrix");
	shaderProg->addUniform("NormalMatrix");
	shaderProg->addUniform("ProjectionMatrix");
	shaderProg->addUniform("SQRotMatrix");
	

	shaderProg->addUniform("Transform");
	shaderProg->addUniform("Scale");
	shaderProg->addUniform("Bright");
}

void SQRenderable::init()
{
	GlyphRenderable::init();
	LoadShaders(glProg);
	//m_vao = std::make_shared<QOpenGLVertexArrayObject>();
	//m_vao->create();

	//m_vao->bind();

	qgl->glGenBuffers(1, &vbo_vert);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 4, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float) * 4, &verts[0].x, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexPosition"));

	qgl->glGenBuffers(1, &vbo_normals);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_normals);
	qgl->glVertexAttribPointer(glProg->attribute("VertexNormal"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(float) * 3, &normals[0].x, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexNormal"));

	qgl->glGenBuffers(1, &vbo_indices);
	qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
	qgl->glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices.size(), &indices[0], GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	initPickingDrawingObjects();

	//m_vao->release();
}

void SQRenderable::DrawWithoutProgram(float modelview[16], float projection[16], ShaderProgram* sp)
{

	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 4, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexPosition"));

	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_normals);
	qgl->glVertexAttribPointer(glProg->attribute("VertexNormal"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexNormal"));

	qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);

	int firstVertex = 0;
	int firstIndex = 0;

	for (int i = 0; i < particle->pos.size(); i++) {
		glPushMatrix();
		//m_vao->bind();

		QMatrix4x4 q_modelview = QMatrix4x4(modelview);
		q_modelview = q_modelview.transposed();

		float3 cen = actor->DataCenter();
		qgl->glUniform4f(glProg->uniform("LightPosition"), 0, 0, std::max(std::max(cen.x, cen.y), cen.z) * 2, 1);
		
		/*if (i == snappedGlyphId)
			qgl->glUniform3f(glProg->uniform("Ka"), 1.0f, 1.0f, 0.3f);
		else if (isFreezingFeature && snappedFeatureId <= 0 && feature[i] > 0){
			qgl->glUniform3f(glProg->uniform("Ka"), 1.0f, 0.3f, 1.0f);
		}
		else if (snappedFeatureId>0 && feature[i] > 0){
			int isFeature = (feature[i] / ((int)pow(2, snappedFeatureId - 1))) % 2;
			if (isFeature)
				qgl->glUniform3f(glProg->uniform("Ka"), 0.3f, 1.0f, 1.0f);
			else
				qgl->glUniform3f(glProg->uniform("Ka"), 0.8f, 0.8f, 0.8f);
			//float kar = 0.3f, kag = 0.3f, kab = 0.3f;
			//if (feature[i] == 1)
			//	kab = 1.0f;
			//else if (feature[i] == 2)
			//	kag = 1.0f;
			//else if (feature[i] == 3)
			//	kar = 1.0f;
			//if (feature[i] == snappedFeatureId){
			//	kar += 0.3f, kag += 0.3f, kab += 0.3f;
			//}
			//qgl->glUniform3f(glProg->uniform("Ka"), kar, kag, kab);
		}
		else*/
			qgl->glUniform3f(glProg->uniform("Ka"), 0.8f, 0.8f, 0.8f);


		qgl->glUniform3f(glProg->uniform("Kd"), 0.3f, 0.3f, 0.3f);
		qgl->glUniform3f(glProg->uniform("Ks"), 0.2f, 0.2f, 0.2f);
		qgl->glUniform1f(glProg->uniform("Shininess"), 1);

		qgl->glUniform1f(glProg->uniform("Bright"), particle->glyphBright[i]);
		qgl->glUniform3fv(glProg->uniform("Transform"), 1, &particle->pos[i].x);

		float glyphSizeAdjust = 1.0f;//glyphSizeAdjust is used when a particle is picked or highlighted, change its size?
		qgl->glUniform1f(glProg->uniform("Scale"), particle->glyphSizeScale[i] * (1 - glyphSizeAdjust) + glyphSizeAdjust);// 1);///*sphereSize[i] * */glyphSizeScale[i]);
		//the data() returns array in column major, so there is no need to do transpose.
		qgl->glUniformMatrix4fv(glProg->uniform("ModelViewMatrix"), 1, GL_FALSE, q_modelview.data());
		qgl->glUniformMatrix4fv(glProg->uniform("ProjectionMatrix"), 1, GL_FALSE, projection);
		//TODO: Not entirely sure about the correctness of the normal matrix, but it works
		qgl->glUniformMatrix3fv(glProg->uniform("NormalMatrix"), 1, GL_FALSE, (q_modelview * rotations[i]).normalMatrix().data());
		qgl->glUniformMatrix4fv(glProg->uniform("SQRotMatrix"), 1, GL_FALSE, rotations[i].data());

		qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
		qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 4, GL_FLOAT,
			GL_FALSE, sizeof(float4), (char*)NULL + firstVertex * sizeof(float4));
		qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_normals);
		qgl->glVertexAttribPointer(glProg->attribute("VertexNormal"), 3, GL_FLOAT,
			GL_TRUE, sizeof(float3), (char*)NULL + firstVertex * sizeof(float3));
		qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);

		glDrawElements(GL_TRIANGLE_STRIP, nIndices[i], GL_UNSIGNED_INT, (char*)NULL + firstIndex * sizeof(unsigned int));

		//m_vao->release();
		glPopMatrix();

		firstVertex += nVerts[i];
		firstIndex += nIndices[i];
	}
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void SQRenderable::draw(float modelview[16], float projection[16])
{

	RecordMatrix(modelview, projection);

	if (!visible)
		return;

	glProg->use();
	DrawWithoutProgram(modelview, projection, glProg);
	glProg->disable();

}

void SQRenderable::UpdateData()
{

}

SQRenderable::SQRenderable(std::shared_ptr<Particle> p) :
GlyphRenderable(p)
//SQRenderable::SQRenderable(vector<float4> _pos, vector<float> _val) :
//GlyphRenderable(_pos)
{
	//deal with it later
	//sphereColor.assign(particle->numParticles, make_float3(1.0f, 1.0f, 1.0f));
	//setColorMap(COLOR_MAP::RDYIGN);

	/* input variables */
	for (int i = 0; i < p->pos.size(); i++) {
		double ten[7] = { p->valTuple[7 * i], p->valTuple[7 * i + 1], p->valTuple[7 * i + 2],
			p->valTuple[7 * i + 3], p->valTuple[7 * i + 4], p->valTuple[7 * i + 5], p->valTuple[7 * i + 6] }; /* tensor coefficients */
		double eps = 1e-4; /* small value >0; defines the smallest tensor
						   * norm at which tensor orientation is still meaningful */

		/* example code starts here */
		double evals[3], evecs[9], uv[2], abc[3], norm;

		tenEigensolve_d(evals, evecs, ten);
		tenGlyphBqdUvEval(uv, evals);
		tenGlyphBqdAbcUv(abc, uv, 3.0);
		norm = ELL_3V_LEN(evals);
		if (norm<eps) {
			double weight = norm / eps;
			abc[0] = weight*abc[0] + (1 - weight);
			abc[1] = weight*abc[1] + (1 - weight);
			abc[2] = weight*abc[2] + (1 - weight);
		}

		/* input variable */
		int glyphRes = 20; /* controls how fine the tesselation will be */

		/* example code starts here */
		limnPolyData *lpd = limnPolyDataNew();
		limnPolyDataSpiralBetterquadric(lpd, (1 << limnPolyDataInfoNorm),
			abc[0], abc[1], abc[2], 0.0,
			2 * glyphRes, glyphRes);
		limnPolyDataVertexNormals(lpd);

		for (int j = 0; j < lpd->xyzwNum; j++) {
			verts.push_back(make_float4(lpd->xyzw[4 * j], 
				lpd->xyzw[4 * j + 1], lpd->xyzw[4 * j + 2], lpd->xyzw[4 * j + 3]));
			normals.push_back(make_float3(lpd->norm[3 * j],
				lpd->norm[3 * j + 1], lpd->norm[3 * j + 2]));
		}
		nVerts.push_back(lpd->xyzwNum);

		for (int j = 0; j < lpd->indxNum; j++) {
			indices.push_back(lpd->indx[j]);
		}
		nIndices.push_back(lpd->indxNum);

		double absevals[3];
		for (int k = 0; k<3; k++)
			absevals[k] = fabs(evals[k]);
		double trans[16] = { absevals[0] * evecs[0], absevals[1] * evecs[3],
			absevals[2] * evecs[6], 0,
			absevals[0] * evecs[1], absevals[1] * evecs[4],
			absevals[2] * evecs[7], 0,
			absevals[0] * evecs[2], absevals[1] * evecs[5],
			absevals[2] * evecs[8], 0,
			0, 0, 0, 1 };

		unsigned int zone = tenGlyphBqdZoneUv(uv);
		if (0 == zone || 5 == zone || 6 == zone || 7 == zone || 8 == zone) {
			/* we need an additional rotation */
			double ZtoX[16] = { 0, 0, 1, 0,
				0, 1, 0, 0,
				-1, 0, 0, 0,
				0, 0, 0, 1 };
			ell_4m_mul_d(trans, trans, ZtoX);
		}
		double gltrans[16];
		ELL_4M_TRANSPOSE(gltrans, trans); /* OpenGL expects column-major format */
		QMatrix4x4 rot = QMatrix4x4(
			gltrans[0], gltrans[4], gltrans[8], gltrans[12], 
			gltrans[1], gltrans[5], gltrans[9], gltrans[13], 
			gltrans[2], gltrans[6], gltrans[10], gltrans[14], 
			gltrans[3], gltrans[7], gltrans[11], gltrans[15]);
		rotations.push_back(rot);
	}
}

void SQRenderable::initPickingDrawingObjects()
{/*
#define GLSL(shader) "#version 150\n" #shader
	//shader is from https://www.packtpub.com/books/content/basics-glsl-40-shaders


	const char* vertexVS =
		GLSL(
		in vec4 VertexPosition;
		uniform mat4 ModelViewMatrix;
		uniform mat4 ProjectionMatrix;
		uniform mat4 SQRotMatrix;
		uniform float Scale;

		uniform vec3 Transform;

		vec4 DivZ(vec4 v){
			return vec4(v.x / v.w, v.y / v.w, v.z / v.w, 1.0f);
		}
		void main()
		{
			mat4 MVP = ProjectionMatrix * ModelViewMatrix;
			gl_Position = MVP * vec4(vec3(DivZ(SQRotMatrix * VertexPosition)) * 1000 * Scale + Transform, 1.0);
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
	glPickingProg->addUniform("SQRotMatrix");
	glPickingProg->addUniform("Scale");

	qgl->glGenBuffers(1, &vbo_vert_picking);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert_picking);
	qgl->glVertexAttribPointer(glPickingProg->attribute("VertexPosition"), 4, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float)* 4, &verts[0].x, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glPickingProg->attribute("VertexPosition"));

	qgl->glGenBuffers(1, &vbo_indices_picking);
	qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices_picking);
	qgl->glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)* indices.size(), &indices[0], GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	*/
}

void SQRenderable::drawPicking(float modelview[16], float projection[16], bool isForGlyph)
{/*
	RecordMatrix(modelview, projection);

	glPickingProg->use();
	
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert_picking);
	qgl->glVertexAttribPointer(glPickingProg->attribute("VertexPosition"), 4, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glEnableVertexAttribArray(glPickingProg->attribute("VertexPosition"));

	qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices_picking);

	int firstVertex = 0;
	int firstIndex = 0;

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

		QMatrix4x4 q_modelview = QMatrix4x4(modelview);
		q_modelview = q_modelview.transposed();

		qgl->glUniform3fv(glPickingProg->uniform("Transform"), 1, &pos[i].x);
		qgl->glUniform1f(glPickingProg->uniform("Scale"), glyphSizeScale[i] * (1 - glyphSizeAdjust) + glyphSizeAdjust);
		//the data() returns array in column major, so there is no need to do transpose.
		qgl->glUniformMatrix4fv(glPickingProg->uniform("ModelViewMatrix"), 1, GL_FALSE, q_modelview.data());
		qgl->glUniformMatrix4fv(glPickingProg->uniform("ProjectionMatrix"), 1, GL_FALSE, projection);
		qgl->glUniformMatrix4fv(glPickingProg->uniform("SQRotMatrix"), 1, GL_FALSE, rotations[i].data());

		qgl->glUniform1f(glPickingProg->uniform("r"), r / 255.0f);
		qgl->glUniform1f(glPickingProg->uniform("g"), g / 255.0f);
		qgl->glUniform1f(glPickingProg->uniform("b"), b / 255.0f);

		qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert_picking);
		qgl->glVertexAttribPointer(glPickingProg->attribute("VertexPosition"), 4, GL_FLOAT,
			GL_FALSE, sizeof(float4), (char*)NULL + firstVertex * sizeof(float4));

		qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices_picking);

		glDrawElements(GL_TRIANGLE_STRIP, nIndices[i], GL_UNSIGNED_INT, (char*)NULL + firstIndex * sizeof(unsigned int));

		//glPopMatrix();

		firstVertex += nVerts[i];
		firstIndex += nIndices[i];
	}


	glPickingProg->disable();
	*/
}