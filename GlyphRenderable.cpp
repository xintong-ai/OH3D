#include "GlyphRenderable.h"
#include <Displace.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <glwidget.h>
#include <Lens.h>
#include <LensRenderable.h>


#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
using namespace std;

#include <QOpenGLFunctions>
#include "ShaderProgram.h"

void GlyphRenderable::ComputeDisplace()
{
	int2 winSize = actor->GetWindowSize();
	displace->Compute(&matrix_mv.v[0].x, &matrix_pj.v[0].x, winSize.x, winSize.y,
		((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses(), &pos[0], &glyphSizeScale[0], &glyphBright[0]);
}


GlyphRenderable::GlyphRenderable(std::vector<float4>& _pos)
{ 
	pos = _pos; 
	displace = std::make_shared<Displace>();
	displace->LoadOrig(&pos[0], pos.size());
	glyphSizeScale.assign(pos.size(), 1.0f);
	glyphBright.assign(pos.size(), 1.0f);
}

GlyphRenderable::~GlyphRenderable()
{ 
	if (nullptr != glProg){
		delete glProg;
		glProg = nullptr;
	}
}


void GlyphRenderable::RecomputeTarget()
{
	displace->RecomputeTarget();
}

void GlyphRenderable::DisplacePoints(std::vector<float2>& pts)
{
	int2 winSize = actor->GetWindowSize();
	displace->DisplacePoints(pts,
		((LensRenderable*)actor->GetRenderable("lenses"))->GetLenses(), &matrix_mv.v[0].x, &matrix_pj.v[0].x, winSize.x, winSize.y);
}

void GlyphRenderable::SlotGlyphSizeAdjustChanged(int v)
{
	glyphSizeAdjust = v * 0.1;
	actor->UpdateGL();
}

void GlyphRenderable::mouseMove(int x, int y, int modifier)
{
}

void GlyphRenderable::resize(int width, int height)
{
	if (INTERACT_MODE::TRANSFORMATION == actor->GetInteractMode()) {
		RecomputeTarget();
	}
}

float3 GlyphRenderable::findClosetGlyph(float3 aim)
{
	return displace->findClosetGlyph(aim, snappedGlyphId);

}



void GlyphRenderable::initForPicking(int nv, float* vertex)
{

	//init shader
#define GLSL(shader) "#version 440\n" #shader
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



	//initializeOpenGLFunctions();
	//glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	//glEnable(GL_DEPTH_TEST);


	//init vertex buffer
	qgl->glGenBuffers(1, &vbo_vert_picking);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert_picking);
	qgl->glVertexAttribPointer(glPickingProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glBufferData(GL_ARRAY_BUFFER, nv * sizeof(float)* 3, vertex, GL_STATIC_DRAW);
	qgl->glBindBuffer(GL_ARRAY_BUFFER, 0);
	qgl->glEnableVertexAttribArray(glPickingProg->attribute("VertexPosition"));

	numVerticeOfGlyph = nv;



	qgl->glGenRenderbuffers(2, renderbuffer);
	qgl->glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer[0]);
	int width, height;
	actor->GetWindowSize(width, height);
	//int width = 750, height = 900;
	width = 900, height = 750;
	qgl->glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, width, height);

	qgl->glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer[1]);
	qgl->glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);


	qgl->glGenFramebuffers(1, &framebuffer);
	qgl->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
	qgl->glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
		GL_RENDERBUFFER, renderbuffer[0]);
	qgl->glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
		GL_RENDERBUFFER, renderbuffer[1]);
}


void GlyphRenderable::drawPicking(float modelview[16], float projection[16])
{

	//std::cout << "draw picking" << std::endl;

	RecordMatrix(modelview, projection);

	ComputeDisplace();
	glPickingProg->use();

	qgl->glBindBuffer(GL_ARRAY_BUFFER, vbo_vert_picking);
	qgl->glVertexAttribPointer(glPickingProg->attribute("VertexPosition"), 3, GL_FLOAT, GL_FALSE, 0, NULL);
	qgl->glEnableVertexAttribArray(glPickingProg->attribute("VertexPosition"));



	for (int i = 0; i < pos.size(); i++) {
		glPushMatrix();

		int r = ((i + 1) & 0x000000FF) >> 0;
		int g = ((i + 1) & 0x0000FF00) >> 8;
		int b = ((i + 1) & 0x00FF0000) >> 16;

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

		
		glDrawArrays(GL_QUADS, 0, numVerticeOfGlyph);
		//glDrawArrays(GL_QUADS, 0, glyphMesh->GetNumVerts());
		//glDrawElements(GL_TRIANGLES, glyphMesh->numElements, GL_UNSIGNED_INT, glyphMesh->indices);
		//m_vao->release();
		glPopMatrix();
	}


	glPickingProg->disable();

}

void GlyphRenderable::mousePress(int x, int y, int modifier)
{
	
	if (QApplication::keyboardModifiers() == Qt::AltModifier && isPicking){
		xMouse = x*1.0/662*750;
		yMouse = y;
//		isPicking = true;

		qgl->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//drawPicking(modelview, projection);
		//drawPicking(&matrix_mv.v[0].x, &matrix_pj.v[0].x);
		drawPicking(mymv, myp);

		qgl->glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
		unsigned char cursorPixel[4];
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		std::cout << "in glrenderable, x and y: " << xMouse << " " << yMouse << std::endl;

		glReadPixels(xMouse, yMouse, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, cursorPixel);
		std::cout << "cursor color: " << (int)cursorPixel[0] << " " << (int)cursorPixel[1] << " "
			<< (int)cursorPixel[2] << " " << (int)cursorPixel[3] << " " << std::endl;

		snappedGlyphId = cursorPixel[0] + cursorPixel[1] * 256 + cursorPixel[2] * 256 * 256 - 1;
		std::cout << "pick id in glyphrenderable: " << snappedGlyphId << std::endl;
	}
	
}