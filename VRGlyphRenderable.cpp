#include <VRGlyphRenderable.h>
#include <GlyphRenderable.h>

#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>

//removing the following lines will cause runtime error
#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
#include "ShaderProgram.h"
void VRGlyphRenderable::init()
{
	glProg = new ShaderProgram;
	glyphRenderable->LoadShaders(glProg);
}

void VRGlyphRenderable::draw(float modelview[16], float projection[16])
{
	if (!visible) 
		return;
	RecordMatrix(modelview, projection);
	glColor3d(1, 1, 1);
	glMatrixMode(GL_MODELVIEW);
	glProg->use();
	glColor3d(1, 0.3, 1);
	glyphRenderable->DrawWithoutProgram(modelview, projection, QOpenGLContext::currentContext(), glProg);
	glProg->disable();
}
