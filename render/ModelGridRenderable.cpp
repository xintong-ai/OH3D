#include <ModelGridRenderable.h>
#include <LineSplitModelGrid.h>
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
//using namespace std;

ModelGridRenderable::ModelGridRenderable(ModelGrid* _modelGrid)//ModelGridRenderable(float dmin[3], float dmax[3], int nPart)
{
	;
}

ModelGridRenderable::ModelGridRenderable(LineSplitModelGrid* _modelGrid)//ModelGridRenderable(float dmin[3], float dmax[3], int nPart)
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

	int3 nStep = modelGrid->GetNumSteps();
	int cutY = nStep.y / 2;

	float* lx = modelGrid->GetX();
	unsigned int* l = modelGrid->GetL();
	float* e = modelGrid->GetE();
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glLineWidth(2);
	glBegin(GL_LINES);
	for (int i = 0; i < modelGrid->GetLNumber(); i++){
		float cc = (e[i / 6] -500)/ 1500;
		glColor4f(cc, 1.0f-cc, 0, 0.5);
		float ccc = clamp(cc, 0.0, 0.5);
		glColor4f(ccc, 1.0f - ccc, 0, 0.5);

		//glColor3f(1.0f, 0, 0);

		glVertex3fv(lx + 3 * l[i * 2]);
		glVertex3fv(lx + 3 * l[i * 2 + 1]);
	}
	glEnd();
	glDisable(GL_BLEND);

	//glColor4f(0.0f, 1.0f, 0, 0.5);
	//glBegin(GL_LINES);
	//int x1, y1, z1;
	//int x2, y2, z2;
	//for (int i = 0; i < modelGrid->GetLNumber(); i++){
	//	
	//	int id1 = l[i * 2], id2 = l[i * 2 + 1];
	//	if (id1 < nStep.x * nStep.y * nStep.z){
	//		x1 = id1 / (nStep.y * nStep.z);
	//		y1 = (id1 - x1* nStep.y * nStep.z) / nStep.z;
	//		z1 = id1 - x1 * nStep.y * nStep.z - y1 * nStep.z;
	//	}
	//	else{
	//		int extra = id1 - nStep.x * nStep.y * nStep.z;
	//		y1 = nStep.y / 2; // always == cutY
	//		z1 = extra / (nStep.x - 2);
	//		x1 = extra - z1*(nStep.x) + 1;
	//	}
	//	if (id2 < nStep.x * nStep.y * nStep.z){
	//		x2 = id2 / (nStep.y * nStep.z);
	//		y2 = (id2 - x2* nStep.y * nStep.z) / nStep.z;
	//		z2 = id2 - x2 * nStep.y * nStep.z - y2 * nStep.z;
	//	}
	//	else{
	//		int extra = id2 - nStep.x * nStep.y * nStep.z;
	//		y2 = nStep.y / 2; // always == cutY
	//		z2 = extra / (nStep.x - 2);
	//		x2 = extra - z2*(nStep.x) + 1;
	//	}

	//	if (y1 >= cutY && y2 <= cutY + 1 && y2 >= cutY && y1 <= cutY + 1 && z1 >= nStep.z - 2 && z2 >= nStep.z - 2){
	//			glVertex3fv(lx + 3 * l[i * 2]);
	//			glVertex3fv(lx + 3 * l[i * 2 + 1]);
	//	}
	//}
	//glEnd();
	//glDisable(GL_BLEND);



	glPointSize(6.0);

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
	for (int i = 0; i < (nStep.x - 2)*nStep.z; i++){

		int idx = nStep.x * nStep.y * nStep.z + i;
			glVertex3fv(lx + 3 * idx);

		
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