#include "LensRenderable.h"
#include "Lens.h"
#include "glwidget.h"

void LensRenderable::init()
{

}

void LensRenderable::UpdateData() {

}

void LensRenderable::draw(float modelview[16], float projection[16])
{
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, winWidth - 1, 0.0, winHeight - 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);
	glLineWidth(4);
	glColor3f(1.0f, 0.2f, 0.2f);
	for (int i = 0; i < lenses.size(); i++) {
		Lens* l = lenses[i];
		std::vector<float2> lensContour = l->GetContour();
		glBegin(GL_LINE_LOOP);
		for (auto v : lensContour)
			glVertex2f(v.x, v.y);
		glEnd();
	}
	glPopAttrib();

	//restore the original 3D coordinate system
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}
//
//void Displace::AddSphereLens(int x, int y, int radius, float3 center)
//{
//	Lens* l = new CircleLens(x, y, radius, center);
//	lenses.push_back(l);
//}


void LensRenderable::AddCircleLens()
{
	Lens* l = new CircleLens(winWidth * 0.5, winHeight * 0.5, winHeight * 0.2, actor->DataCenter());
	lenses.push_back(l);
	actor->UpdateGL();
}