#include <ModelGridRenderable.h>
#include <ModelGrid.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>

#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
#include <QOpenGLFunctions>
#include <LensRenderable.h>
#include <glwidget.h>
using namespace std;


ModelGridRenderable::ModelGridRenderable(ModelGrid* _modelGrid)//ModelGridRenderable(float dmin[3], float dmax[3], int nPart)
{
	modelGrid = _modelGrid;
//	modelGrid = new ModelGrid(dmin, dmax, nPart);
	visible = false;
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

	qgl->glUseProgram(0);
	RecordMatrix(modelview, projection);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


	if (!visible)
		return;
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(modelview);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(projection);

	//glEnableClientState(GL_VERTEX_ARRAY);
	//glVertexPointer(3, GL_FLOAT, 0, modelGrid->GetX());

	//glDrawElements(GL_LINES, modelGrid->GetLNumber() * 2, GL_UNSIGNED_INT, modelGrid->GetL());
	//glDisableClientState(GL_VERTEX_ARRAY);

	float* lx = modelGrid->GetX();
	unsigned int* l = modelGrid->GetL();
	float* e = modelGrid->GetE();
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glBegin(GL_LINES);
	for (int i = 0; i < modelGrid->GetLNumber(); i++){
		float cc = e[i / 6] / 10000;
		glColor4f(cc, 1.0f-cc, 0, 0.5);
		//glColor3f(1.0f, 0, 0);

		glVertex3fv(lx + 3 * l[i * 2]);
		glVertex3fv(lx + 3 * l[i * 2 + 1]);
	}
	glEnd();
	glDisable(GL_BLEND);

	int3 nStep = modelGrid->GetNumSteps();
	int cutY = nStep.y / 2, cutY2 = cutY + 1;

	glPointSize(5.0);

	glColor3f(0.0f, 0.0f, 1.0f);
	glBegin(GL_POINTS);
	for (int i = 0; i < nStep.x; i++){
		for (int k = 0; k < nStep.z; k++)
		{
			int idx = i * nStep.y * nStep.z + cutY * nStep.z + k;
			glVertex3fv(lx + 3 * idx);

		}
	}
	glEnd();

	glColor3f(1.0f, 1.0f, 0.0f);
	glBegin(GL_POINTS);
	for (int i = 0; i < nStep.x; i++){
		for (int k = 0; k < nStep.z; k++)
		{
			int idx = i * nStep.y * nStep.z + (cutY+1) * nStep.z + k;
			glVertex3fv(lx + 3 * idx);

		}
	}
	glEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void ModelGridRenderable::mouseRelease(int x, int y, int modifier)
{
	modelGrid->setReinitiationNeed();
}