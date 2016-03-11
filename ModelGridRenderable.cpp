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
ModelGridRenderable::ModelGridRenderable(float dmin[3], float dmax[3], int nPart)
{
	modelGrid = new ModelGrid(dmin, dmax, nPart);
}

void ModelGridRenderable::UpdateGridDensity(float4* v, int n)
{
	//;
	float3 gridMin = modelGrid->GetGridMin();
	float3 gridMax = modelGrid->GetGridMax();
	int3 nStep = modelGrid->GetNumSteps();
	float step = modelGrid->GetStep();
	std::vector<int> cnts;
	cnts.resize((nStep.x - 1) *(nStep.y - 1) *(nStep.z - 1), 0);
	for (int i = 0; i < n; i++){
		float3 vc = make_float3(v[i].x, v[i].y, v[i].z);
		float3 tmp = (vc - gridMin) / step;
		int3 idx3 = make_int3(tmp.x, tmp.y, tmp.z);
		int idx = idx3.x * (nStep.y - 1) * (nStep.z - 1)
			+ idx3.y * (nStep.z - 1) + idx3.z;
		float3 local = make_float3(tmp.x - idx3.x, tmp.y - idx3.y, tmp.z - idx3.z);// vc - (gridMin + make_float3(idx3.x * step, idx3.y * step, idx3.z * step));
		localCoord.push_back(make_float4(idx, local.x, local.y, local.z));
		cnts[idx]++;
	}
	std::vector<float> density;
	density.resize(cnts.size() * 5);
	const float base = 400.0f / cnts.size();
	for (int i = 0; i < cnts.size(); i++) {
		for (int j = 0; j < 5; j++) {
			density[i * 5 + j] = 1800 + 10000 * (float)cnts[i] ;
		}
	}
	modelGrid->SetElasticity(&density[0]);
	modelGrid->Initialize(time_step);
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
	float3 lensCen = ((LensRenderable*)actor->GetRenderable("lenses"))->GetBackLensCenter();

	modelGrid->Update(time_step, &lensCen.x);
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
	glBegin(GL_LINES);
	for (int i = 0; i < modelGrid->GetLNumber(); i++){
		glColor3f(e[i / 6] / 100000, 0, 0);
		//glColor3f(1.0f, 0, 0);

		glVertex3fv(lx + 3 * l[i * 2]);
		glVertex3fv(lx + 3 * l[i * 2 + 1]);
	}
	glEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}
