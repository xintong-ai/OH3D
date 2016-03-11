#include <ModelGridRenderable.h>
#include <ModelGrid.h>
#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
#include <QOpenGLFunctions>
using namespace std;
ModelGridRenderable::ModelGridRenderable(float dmin[3], float dmax[3], int nPart)
{
	modelGrid = new ModelGrid(dmin, dmax, nPart);
}

void ModelGridRenderable::init()
{
	//Create VBO
	//qgl->glGenBuffers(1, &vertex_handle);
	//qgl->glGenBuffers(1, &triangle_handle);

	//qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_handle);
	//qgl->glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*modelGrid->GetTNumber() * 3, modelGrid->GetT(), GL_STATIC_DRAW);
}

void ModelGridRenderable::draw(float modelview[16], float projection[16])
{
	if (!visible)
		return;
	qgl->glUseProgram(0);
	RecordMatrix(modelview, projection);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(projection);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(modelview);
	//glBegin(GL_TRIANGLES);
	//glVertex3f(-1, 1, 0);
	//glVertex3f(-1, 0, 1);
	//glVertex3f(1, 1, 0);
	//glEnd();

	//GLint vertices[] =
	//{
	//	-1, 1, 0,
	//	-1, 0, 1,
	//	1, 1, 0,
	//	0, 1, 0,
	//	-1, 1, 0,
	//	-1, 0, 0,
	//};

	//GLubyte colors[] =
	//{
	//	255, 0, 0,
	//	0, 255, 0,
	//	0, 0, 255,
	//	255, 255, 0,
	//	255, 0, 255,
	//	0, 255, 255,
	//};

	//GLubyte indices[] =
	//{
	//	0, 1, 2,
	//	0, 3, 4,
	//};

	//glColor3f(1.0f, 1.0f, 1.0f);
	////glEnableClientState(GL_COLOR_ARRAY);
	//glEnableClientState(GL_VERTEX_ARRAY);
	////glColorPointer(3, GL_UNSIGNED_BYTE, 0, colors);
	//glVertexPointer(3, GL_INT, 0, vertices);
	//glDrawElements(GL_LINES, sizeof(indices) / sizeof(GLubyte), GL_UNSIGNED_BYTE, indices);

	//GLfloat vertices[] =
	//{
	//	0, 0, 0,
	//	1, 0, 0,
	//	1, 1, 0,
	//	0, 1, 0,
	//	-1, 1, 0,
	//	-1, 0, 0,
	//};

	//unsigned int indices[] =
	//{
	//	0, 1, 2, 3,
	//	0, 3, 4, 5,
	//};

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, modelGrid->GetX());

	glDrawElements(GL_LINES, modelGrid->GetLNumber() * 2, GL_UNSIGNED_INT, modelGrid->GetL());
	glDisableClientState(GL_VERTEX_ARRAY);

	//qgl->glBindBuffer(GL_ARRAY_BUFFER, vertex_handle);
	//qgl->glBufferData(GL_ARRAY_BUFFER, sizeof(float)*modelGrid->GetNumber() * 3, modelGrid->GetX(), GL_DYNAMIC_DRAW);
	//qgl->glEnableVertexAttribArray(0);    //We like submitting vertices on stream 0 for no special reason
	//qgl->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), NULL);   //The starting point of the VBO, for the vertices

	//qgl->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_handle);
	////To render, we can either use glDrawElements or glDrawRangeElements
	////The is the number of indices. 3 indices needed to make a single triangle
	//glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_SHORT, NULL);   //The starting point of the IBO

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
