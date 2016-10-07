#include "ArrowNoDeformRenderable.h"
//#include <teem/ten.h>
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
#include "helper_math.h"
#include "GLArrow.h"
#include "ColorGradient.h"

using namespace std;


inline
QMatrix4x4 ArrowNoDeformRenderable::computeRotationMatrix(float3 orientation, float3 _vec)
{
	float3 norVec = normalize(_vec);
	float sinTheta = length(cross(orientation, norVec));
	float cosTheta = dot(orientation, norVec);

	if (sinTheta<0.00001)
	{
		return QMatrix4x4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}
	else
	{
		float3 axis = normalize(cross(orientation, norVec));
		return QMatrix4x4(
			cosTheta + axis.x*axis.x*(1 - cosTheta), axis.x*axis.y*(1 - cosTheta) - axis.z*sinTheta,
			axis.x*axis.z*(1 - cosTheta) + axis.y*sinTheta, 0,
			axis.y*axis.x*(1 - cosTheta) + axis.z*sinTheta, cosTheta + axis.y*axis.y*(1 - cosTheta),
			axis.y*axis.z*(1 - cosTheta) - axis.x*sinTheta, 0,
			axis.z*axis.x*(1 - cosTheta) - axis.y*sinTheta, axis.z*axis.y*(1 - cosTheta) + axis.x*sinTheta,
			cosTheta + axis.z*axis.z*(1 - cosTheta), 0,
			0, 0, 0, 1);
	}
}


ArrowNoDeformRenderable::ArrowNoDeformRenderable(vector<float3> _vec, std::shared_ptr<Particle> _particle) :
GlyphRenderable(_particle)
{
	cout << "ArrowNoDeformRenderable started" << endl;

	if (particle->numParticles != _vec.size()){
		cerr << "ArrowNoDeformRenderable constructing fault";
		exit(0);
	}

	sphereColor.assign(particle->numParticles, make_float3(1.0f, 1.0f, 1.0f));

	vecs = _vec;

	lMax = -1, lMin = 999999;
	glyphMesh = std::make_shared<GLArrow>();
	float3 orientation = glyphMesh->orientation;

	for (int i = 0; i < vecs.size(); i++) {

		float l = length(_vec[i]);
		if (l>lMax)
			lMax = l;
		if (l < lMin)
			lMin = l;

		rotations.push_back(computeRotationMatrix(orientation, _vec[i]));
	}

	setColorMap(COLOR_MAP::SIMPLE_BLUE_RED);
	cout << "ArrowNoDeformRenderable created" << endl;
}


void ArrowNoDeformRenderable::setColorMap(COLOR_MAP cm, bool isReversed)
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


void ArrowNoDeformRenderable::LoadShaders(ShaderProgram*& shaderProg)
{

#define GLSL(shader) "#version 150\n" #shader
	//shader is from https://www.packtpub.com/books/content/basics-glsl-40-shaders


	const char* vertexVS =
		GLSL(
		in vec4 VertexPosition;
	in vec4 VertexColor;
	in vec3 VertexNormal;
	smooth out vec3 tnorm;
	out vec4 eyeCoords;
	out vec4 fragColor;

	uniform mat4 ModelViewMatrix;
	uniform mat3 NormalMatrix;
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
		eyeCoords = ModelViewMatrix * VertexPosition;
		tnorm = normalize(NormalMatrix * /*vec3(VertexPosition) + 0.001 * */VertexNormal);
		//gl_Position = MVP * (VertexPosition + vec4(Transform, 0.0));
		if (Scale<1)
			gl_Position = MVP * vec4(vec3(DivZ(SQRotMatrix * (VertexPosition*vec4(Scale, Scale, Scale, 1.0)))) + Transform, 1.0);
		else
			gl_Position = MVP * vec4(vec3(DivZ(SQRotMatrix * (VertexPosition*vec4(1.0, 1.0, Scale, 1.0)))) + Transform, 1.0);
		fragColor = VertexColor;
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
	in vec4 fragColor;

	uniform float Bright;

	smooth in vec3 tnorm;
	//layout(location = 0) out vec4 FragColor;
	out vec4 FragColor;

	vec3 phongModel(vec3 a, vec4 position, vec3 normal) {
		vec3 s = normalize(vec3(LightPosition - position));
		vec3 v = normalize(-position.xyz);
		vec3 r = reflect(-s, normal);
		vec3 ambient = a + vec3(fragColor)*0.00001;// Ka * 0.8;
		float sDotN = max(dot(s, normal), 0.0);
		vec3 diffuse = Kd * sDotN;
		vec3 spec = vec3(0.0);
		if (sDotN > 0.0)
			spec = Ks *
			pow(max(dot(r, v), 0.0), Shininess);
		return ambient + diffuse + spec;
	}

	void main() {

		FragColor = vec4(Bright * phongModel(Ka * 0.5, eyeCoords, tnorm), 0.0);
	}
	);

	shaderProg = new ShaderProgram;
	shaderProg->initFromStrings(vertexVS, vertexFS);

	shaderProg->addAttribute("VertexPosition");
	shaderProg->addAttribute("VertexColor");
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
	shaderProg->addUniform("Scale");
	shaderProg->addUniform("Bright");
	shaderProg->addUniform("Transform");
}

void ArrowNoDeformRenderable::GenVertexBuffer(int nv, float* vertex)
{
	qgl->glGenBuffers(1, &vbo_vert);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	qgl->glVertexAttribPointer(glProg->attribute("VertexPosition"), 4, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, glyphMesh->GetNumVerts() * sizeof(float)* 4, glyphMesh->GetVerts(), GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexPosition"));

	qgl->glGenBuffers(1, &vbo_normals);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_normals);
	qgl->glVertexAttribPointer(glProg->attribute("VertexNormal"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, glyphMesh->GetNumVerts() * sizeof(float)* 3, glyphMesh->GetNormals(), GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexNormal"));

	qgl->glGenBuffers(1, &vbo_colors);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
	qgl->glVertexAttribPointer(glProg->attribute("VertexColor"), 4, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, glyphMesh->GetNumVerts() * sizeof(float)* 4, glyphMesh->GetColors(), GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glProg->attribute("VertexColor"));

	qgl->glGenBuffers(1, &vbo_indices);
	qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
	qgl->glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)* glyphMesh->GetNumIndices(), glyphMesh->GetIndices(), GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void ArrowNoDeformRenderable::init()
{
	cout << "ArrowNoDeformRenderable init created" << endl;

	GlyphRenderable::init();

	m_vao = std::make_shared<QOpenGLVertexArrayObject>();
	m_vao->create();

	//glyphMesh has been created, and by now must be create during construstion

	m_vao->bind();
	LoadShaders(glProg);

	GenVertexBuffer(glyphMesh->GetNumVerts(), glyphMesh->GetVerts());

	initPickingDrawingObjects();
	m_vao->release();

	cout << "ArrowNoDeformRenderable init finished" << endl;

}

void ArrowNoDeformRenderable::DrawWithoutProgram(float modelview[16], float projection[16], ShaderProgram* sp)
{
	return;
	int firstVertex = 0;
	int firstIndex = 0;
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert);
	qgl->glVertexAttribPointer(sp->attribute("VertexPosition"), 4, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glEnableVertexAttribArray(sp->attribute("VertexPosition"));
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_normals);
	qgl->glVertexAttribPointer(sp->attribute("VertexNormal"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glEnableVertexAttribArray(sp->attribute("VertexNormal"));
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
	qgl->glVertexAttribPointer(sp->attribute("VertexColor"), 4, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glEnableVertexAttribArray(sp->attribute("VertexColor"));
	
	qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);

	for (int i = 0; i < particle->numParticles; i++) {


		QMatrix4x4 q_modelview = QMatrix4x4(modelview);
		q_modelview = q_modelview.transposed();
		float3 cen = actor->DataCenter();
		qgl->glUniform4f(sp->uniform("LightPosition"), 0, 0, std::max(std::max(cen.x, cen.y), cen.z) * 2, 1);

		//float3 vec = vecs[i];
		//float cosX = dot(vec, make_float3(1, 0, 0));
		//float cosY = dot(vec, make_float3(0, 1, 0));
		//qgl->glUniform3f(sp->uniform("Ka"), (cosY + 1) / 2, (cosX + 1) / 2, 1 - (cosX + 1) / 2);
		//qgl->glUniform3f(sp->uniform("Ka"), 0.8f, 0.8f, 0.8f);
		qgl->glUniform3f(sp->uniform("Kd"), 0.3f, 0.3f, 0.3f);
		qgl->glUniform3f(sp->uniform("Ks"), 0.2f, 0.2f, 0.2f);
		qgl->glUniform1f(sp->uniform("Shininess"), 5);
		qgl->glUniformMatrix4fv(sp->uniform("ModelViewMatrix"), 1, GL_FALSE, modelview);
		qgl->glUniformMatrix4fv(sp->uniform("ProjectionMatrix"), 1, GL_FALSE, projection);


		////qgl->glUniformMatrix3fv(sp->uniform("NormalMatrix"), 1, GL_FALSE, q_modelview.normalMatrix().data());
		//qgl->glUniformMatrix3fv(sp->uniform("NormalMatrix"), 1, GL_FALSE, (q_modelview * rotations[i]).normalMatrix().data());
		//qgl->glUniformMatrix4fv(sp->uniform("SQRotMatrix"), 1, GL_FALSE, rotations[i].data());
		//qgl->glUniform3fv(sp->uniform("Transform"), 1, &shift.x);

		
		QVector3D q_IndicatorDir;
		if(i==0)
			q_IndicatorDir = QVector3D(QMatrix4x4((q_modelview.inverted()).normalMatrix()).map(QVector4D(-1, 1, 0, 0))).normalized();
		else
			q_IndicatorDir = QVector3D(QMatrix4x4((q_modelview.inverted()).normalMatrix()).map(QVector4D(1, 1, 0, 0))).normalized();
		float3 curVec = make_float3(q_IndicatorDir.x(), q_IndicatorDir.y(), q_IndicatorDir.z());
		
		QMatrix4x4 curRotation = computeRotationMatrix(glyphMesh->orientation, curVec);
		qgl->glUniformMatrix3fv(sp->uniform("NormalMatrix"), 1, GL_FALSE, (q_modelview * curRotation).normalMatrix().data());
		qgl->glUniformMatrix4fv(sp->uniform("SQRotMatrix"), 1, GL_FALSE, curRotation.data());

		float4 shift = pos[i] - make_float4(curVec / 2.0, 0.0);
		qgl->glUniform3fv(sp->uniform("Transform"), 1, &shift.x);


		float maxScaleInv = 8;
		qgl->glUniform1f(sp->uniform("Bright"), glyphBright[i]);
		qgl->glUniform1f(sp->uniform("Scale"), length(vecs[i]));

		if (particle->val[i] == 0){
			sphereColor[i].x = 1.0;
			sphereColor[i].y = 0.0;
			sphereColor[i].z = 0.0;
		}
		else{
			sphereColor[i].x = 1.0;
			sphereColor[i].y = 1.0;
			sphereColor[i].z = 1.0;
		}

		if (i != snappedGlyphId){
			qgl->glUniform3fv(sp->uniform("Ka"), 1, &sphereColor[i].x);
		}
		else{
			qgl->glUniform3f(sp->uniform("Ka"), 0.9f, 0.9f, 0.9f);
		}
		glColor3f(1, 1, 0);
		glDrawArrays(GL_TRIANGLES, 0, glyphMesh->GetNumVerts());
	}

	qgl->glDisableVertexAttribArray(sp->attribute("VertexPosition"));
	qgl->glDisableVertexAttribArray(sp->attribute("VertexNormal"));
	qgl->glDisableVertexAttribArray(sp->attribute("VertexColor"));
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

}


void ArrowNoDeformRenderable::draw(float modelview[16], float projection[16])
{
	if (!visible)
		return;

	RecordMatrix(modelview, projection);

	//glEnable(GL_BLEND);
	////glBlendFunc(GL_SRC_ALPHA, GL_ZERO);	
	//glBlendFunc(GL_SRC_ALPHA, GL_ZERO);

	glProg->use();
	DrawWithoutProgram(modelview, projection, glProg);
	glProg->disable();

	//glDisable(GL_BLEND);
}


void ArrowNoDeformRenderable::initPickingDrawingObjects()
{

	//init shader
#define GLSL(shader) "#version 150\n" #shader
	//shader is from https://www.packtpub.com/books/content/basics-glsl-40-shaders
	//using two sides shading
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
		//gl_Position = MVP * (VertexPosition + vec4(Transform, 0.0));
		if (Scale<1)
			gl_Position = MVP * vec4(vec3(DivZ(SQRotMatrix * (VertexPosition*vec4(Scale, Scale, Scale, 1.0)))) + Transform, 1.0);
		else
			gl_Position = MVP * vec4(vec3(DivZ(SQRotMatrix * (VertexPosition*vec4(1.0, 1.0, Scale, 1.0)))) + Transform, 1.0);
	}
	);

	const char* vertexFS =
		GLSL(
		//layout(location = 0) out vec4 FragColor;
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
	glPickingProg->addUniform("SQRotMatrix");
	glPickingProg->addUniform("Scale");

	//init vertex buffer
	qgl->glGenBuffers(1, &vbo_vert_picking);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert_picking);
	qgl->glVertexAttribPointer(glPickingProg->attribute("VertexPosition"), 4, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, glyphMesh->GetNumVerts() * sizeof(float)* 4, glyphMesh->GetVerts(), GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glPickingProg->attribute("VertexPosition"));
}


void ArrowNoDeformRenderable::drawPicking(float modelview[16], float projection[16], bool isForGlyph)
{
	RecordMatrix(modelview, projection);

	glPickingProg->use();

	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert_picking);
	qgl->glVertexAttribPointer(glPickingProg->attribute("VertexPosition"), 4, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glEnableVertexAttribArray(glPickingProg->attribute("VertexPosition"));

	for (int i = 0; i < particle->numParticles; i++) {
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

		qgl->glUniform3fv(glPickingProg->uniform("Transform"), 1, &shift.x);
		qgl->glUniformMatrix4fv(glPickingProg->uniform("ModelViewMatrix"), 1, GL_FALSE, modelview);
		qgl->glUniformMatrix4fv(glPickingProg->uniform("ProjectionMatrix"), 1, GL_FALSE, projection);
		qgl->glUniformMatrix4fv(glPickingProg->uniform("SQRotMatrix"), 1, GL_FALSE, rotations[i].data());
		qgl->glUniform1f(glPickingProg->uniform("r"), r / 255.0f);
		qgl->glUniform1f(glPickingProg->uniform("g"), g / 255.0f);
		qgl->glUniform1f(glPickingProg->uniform("b"), b / 255.0f);

		float maxScaleInv = 8;
		qgl->glUniform1f(glPickingProg->uniform("Scale"), length(vecs[i]) / lMax * maxScaleInv);


		glDrawArrays(GL_TRIANGLES, 0, glyphMesh->GetNumVerts());

		//glPopMatrix();
	}

	glPickingProg->disable();
}