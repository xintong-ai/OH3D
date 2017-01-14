#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>

#include "GlyphRenderable.h"
#include "glwidget.h"
#include "DeformGLWidget.h"
#include "Particle.h""


#ifdef WIN32
#include <windows.h>
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
//using namespace std;

#include <QOpenGLFunctions>
#include "ShaderProgram.h"

GlyphRenderable::GlyphRenderable(std::shared_ptr<Particle> _particle)
{
	particle = _particle;
	particle->initForRendering();
}

GlyphRenderable::~GlyphRenderable()
{ 
	if (nullptr != glProg){
		delete glProg;
		glProg = nullptr;
	}
}



void GlyphRenderable::resize(int width, int height)
{
	if (!frameBufferObjectInitialized){
		qgl->glGenRenderbuffers(2, renderbuffer);
		qgl->glGenFramebuffers(1, &framebuffer);
		frameBufferObjectInitialized = true;
	}

	qgl->glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer[0]);
	qgl->glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, width, height);

	qgl->glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer[1]);
	qgl->glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);


	qgl->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
	qgl->glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
		GL_RENDERBUFFER, renderbuffer[0]);
	qgl->glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
		GL_RENDERBUFFER, renderbuffer[1]);
	qgl->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}


