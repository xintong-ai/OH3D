#include "MatrixMgrRenderable.h"
#include "GLMatrixManager.h"
#include "TransformFunc.h"

#include "GLWidget.h"

void MatrixMgrRenderable::draw(float modelview[16], float projection[16])
{
	RecordMatrix(modelview, projection);
	
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(projection);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(modelview);

	glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);
	glLineWidth(4); 

	float _invmv[16];
	float _invpj[16];
	invertMatrix(projection, _invpj);
	invertMatrix(modelview, _invmv);

	int2 winSize = actor->GetWindowSize();
	float2 censcreen = make_float2(0.25*winSize.x, 0.25*winSize.y);
	float4 cen = Clip2ObjectGlobal(make_float4(make_float3(Screen2Clip(censcreen, winSize.x, winSize.y), 0.95), 1.0), _invmv, _invpj);

	//float3 cen = matrixMgr->getEyeInLocal() + 10 * matrixMgr->getViewVecInLocal();
	float l = 2;
	glColor3f(0.90f, 0.10f, 0.10f);
	glBegin(GL_LINES);
	glVertex3f(cen.x, cen.y, cen.z);
	glVertex3f(cen.x + l, cen.y, cen.z);
	glEnd();

	glColor3f(0.90f, 0.90f, 0.10f);
	glBegin(GL_LINES);
	glVertex3f(cen.x, cen.y, cen.z);
	glVertex3f(cen.x, cen.y + l, cen.z);
	glEnd();

	glColor3f(0.10f, 0.90f, 0.10f);
	glBegin(GL_LINES);
	glVertex3f(cen.x, cen.y, cen.z);
	glVertex3f(cen.x, cen.y, cen.z + l);
	glEnd();


	glPopAttrib();
	//restore the original 3D coordinate system
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();


	if (!visible)
		return;
}