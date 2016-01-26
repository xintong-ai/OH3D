#include "SQRenderable.h"
#include <teem/ten.h>
//http://www.sci.utah.edu/~gk/vissym04/index.html
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>

//removing the following lines will cause runtime error
#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
#include "ShaderProgram.h"

#include <memory>
#include "glwidget.h"
using namespace std;

void SQRenderable::LoadShaders()
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
	uniform float Scale;

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
		FragColor = vec4(phongModel(Ka * 0.5, eyeCoords, tnorm), 1.0);
	}
	);

	glProg = std::make_unique<ShaderProgram>();
	glProg->initFromStrings(vertexVS, vertexFS);

	glProg->addAttribute("VertexPosition");
	glProg->addUniform("LightPosition");
	glProg->addUniform("Ka");
	glProg->addUniform("Kd");
	glProg->addUniform("Ks");
	glProg->addUniform("Shininess");

	glProg->addUniform("ModelViewMatrix");
	glProg->addUniform("NormalMatrix");
	glProg->addUniform("ProjectionMatrix");

	glProg->addUniform("Transform");
	glProg->addUniform("Scale");
}

void SQRenderable::init()
{
	LoadShaders();
	m_vao = std::make_unique<QOpenGLVertexArrayObject>();
	m_vao->create();

	m_vao->bind();

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

	m_vao->release();
}
void SQRenderable::draw(float modelview[16], float projection[16])
{
	if (!visible)
		return;

	RecordMatrix(modelview, projection);
	ComputeDisplace();

	glMatrixMode(GL_MODELVIEW);

	int firstVertex = 0;
	int firstIndex = 0;

	for (int i = 0; i < pos.size(); i++) {
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
		qgl->glUniform3f(glProg->uniform("Ka"), 0.8f, 0.8f, 0.8f);
		qgl->glUniform3f(glProg->uniform("Kd"), 0.3f, 0.3f, 0.3f);
		qgl->glUniform3f(glProg->uniform("Ks"), 0.2f, 0.2f, 0.2f);
		qgl->glUniform1f(glProg->uniform("Shininess"), 5);
		qgl->glUniform3fv(glProg->uniform("Transform"), 1, &shift.x);
		qgl->glUniform1f(glProg->uniform("Scale"), glyphSizeScale[i] * (1 - glyphSizeAdjust) + glyphSizeAdjust);// 1);///*sphereSize[i] * */glyphSizeScale[i]);
		qgl->glUniformMatrix4fv(glProg->uniform("ModelViewMatrix"), 1, GL_FALSE, modelview);
		qgl->glUniformMatrix4fv(glProg->uniform("ProjectionMatrix"), 1, GL_FALSE, projection);
		qgl->glUniformMatrix3fv(glProg->uniform("NormalMatrix"), 1, GL_FALSE, q_modelview.normalMatrix().data());

		qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
		qgl->glEnableVertexAttribArray(0);    //We like submitting vertices on stream 0 for no special reason
		qgl->glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (char*)NULL + firstVertex * sizeof(float4));   //The starting point of the VBO, for the vertices
		qgl->glEnableVertexAttribArray(1);    //We like submitting normals on stream 1 for no special reason
		qgl->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (char*)NULL + firstVertex * sizeof(float3));     //The starting point of normals, 12 bytes away
		//qgl->glEnableVertexAttribArray(2);    //We like submitting texcoords on stream 2 for no special reason
		//qgl->glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(MyVertex), BUFFER_OFFSET(24));   //The starting point of texcoords, 24 bytes away

		qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);

		//glDrawArrays(GL_QUADS, 0, glyphMesh->GetNumVerts());
		glDrawElements(GL_TRIANGLE_STRIP, nIndices[i], GL_UNSIGNED_INT, (char*)NULL + firstIndex * sizeof(float3));
		m_vao->release();
		glProg->disable();
		glPopMatrix();

		firstVertex += nVerts[i];
		firstIndex += nIndices[i];
	}

}
void SQRenderable::UpdateData()
{

}

SQRenderable::SQRenderable(vector<float4> _pos, vector<float> _val) :
GlyphRenderable(_pos)
{
	val = _val;
	/* input variables */
	for (int i = 0; i < pos.size(); i++) {
		double ten[7] = { val[7 * i], val[7 * i + 1], val[7 * i + 2], 
			val[7 * i + 3], val[7 * i + 4], val[7 * i + 5], val[7 * i + 6] }; /* tensor coefficients */
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
				lpd->xyzw[3 * j + 1], lpd->xyzw[3 * j + 2]));
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
	}
}
