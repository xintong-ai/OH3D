#include <ModelGridRenderable.h>
#include <LineSplitModelGrid.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include "Lens.h"

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
	
	if ((*lenses).size() < 1)
		return;
	
	
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(modelview);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(projection);

	if (modelGrid->gridType == GRID_TYPE::LINESPLIT_UNIFORM_GRID){
		int3 nStep = modelGrid->GetNumSteps();
		int cutY = nStep.y / 2;

		float* lx = modelGrid->GetX();
		unsigned int* l = modelGrid->GetL();
		float* e = modelGrid->GetE();
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		glLineWidth(2);
		glBegin(GL_LINES);

		float minElas = modelGrid->minElas;
		float maxElas = modelGrid->maxElasEstimate;
		LineLens3D* len = (LineLens3D*)((*lenses)[0]);
		float3 lensCen = len->c;
		float3 lensDir = len->lensDir;

		for (int i = 0; i < modelGrid->GetLNumber(); i++){

			float cc = (e[i / 6] - minElas) / (maxElas - minElas);
			glColor4f(cc, 1.0f-cc, 0, 0.5);

			//glColor4f(0.2f, 0.8f, 0.1, 0.5);

			float *pp1 = lx + 3 * l[i * 2], *pp2 = lx + 3 * l[i * 2 + 1];
			float3 v1 = make_float3(pp1[0], pp1[1], pp1[2]);
			float3 v2 = make_float3(pp2[0], pp2[1], pp2[2]);
			if (dot(v1 - lensCen, lensDir)>0 || dot(v2 - lensCen, lensDir) > 0){
				glVertex3fv(pp1);
				glVertex3fv(pp2);
			}
		}
		glEnd();
		glDisable(GL_BLEND);


		glPointSize(6.0);

		glColor3f(0.0f, 0.0f, 1.0f);
		glBegin(GL_POINTS);
		for (int i = 1; i < nStep.x - 1; i++){
			for (int k = 0; k < nStep.z; k++)
			{
				int idx = i * nStep.y * nStep.z + cutY * nStep.z + k;

				float *pp1 = lx + 3 * idx;
				float3 v1 = make_float3(pp1[0], pp1[1], pp1[2]);
				if (dot(v1 - lensCen, lensDir)>0){
					glVertex3fv(lx + 3 * idx);
				}
			}
		}
		glEnd();

		glColor3f(1.0f, 0.0f, 0.0f);
		glBegin(GL_POINTS);
		for (int i = 0; i < (nStep.x - 2)*nStep.z; i++){

			int idx = nStep.x * nStep.y * nStep.z + i;
			float *pp1 = lx + 3 * idx;
			float3 v1 = make_float3(pp1[0], pp1[1], pp1[2]);
			if (dot(v1 - lensCen, lensDir)>0){
				glVertex3fv(lx + 3 * idx);
			}

		}
		glEnd();
	}
	else if (modelGrid->gridType == GRID_TYPE::UNIFORM_GRID){
		float* lx = modelGrid->GetX();
		unsigned int* l = modelGrid->GetL();
		float* e = modelGrid->GetE();
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glLineWidth(4);

		glBegin(GL_LINES);
		for (int i = 0; i < modelGrid->GetLNumber(); i++){
			//float cc = e[i / 6] / 10000;
			//glColor4f(cc, 1.0f - cc, 0, 0.5);
			glColor4f(0.02f, 0.8f, 0.0f, 0.5f);

			glVertex3fv(lx + 3 * l[i * 2]);
			glVertex3fv(lx + 3 * l[i * 2 + 1]);
		}
		glEnd();
		glDisable(GL_BLEND);
	}
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void ModelGridRenderable::mouseRelease(int x, int y, int modifier)
{
	//modelGrid->setReinitiationNeed();
}