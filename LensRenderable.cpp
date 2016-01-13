#include "LensRenderable.h"
#include "GlyphRenderable.h"
#include "Lens.h"
#include "glwidget.h"

void LensRenderable::init()
{

}

void LensRenderable::UpdateData() {

}

void LensRenderable::draw(float modelview[16], float projection[16])
{
	RecordMatrix(modelview, projection);

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
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
}

void LensRenderable::mousePress(int x, int y, int modifier)
{
	for (int i = 0; i < lenses.size(); i++) {
		Lens* l = lenses[i];
		if (l->PointInsideLens(x, y)) {
			//workingOnLens = true;
			actor->SetInteractMode(INTERACT_MODE::LENS);
			pickedLens = i;
			lastPt = make_int2(x, y);
		}
	}
	//return insideAnyLens;
}

void LensRenderable::mouseRelease(int x, int y, int modifier)
{
	actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
	//workingOnLens = false;
}

void LensRenderable::mouseMove(int x, int y, int modifier)
{
	if (INTERACT_MODE::LENS == actor->GetInteractMode()) {
		lenses[pickedLens]->x += (x - lastPt.x);
		lenses[pickedLens]->y += (y - lastPt.y);
	}
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	lastPt = make_int2(x, y);
}

bool LensRenderable::MouseWheel(int x, int y, int delta)
{
	bool insideAnyLens = false;
	for (int i = 0; i < lenses.size(); i++) {
		Lens* l = lenses[i];
		if (l->PointInsideLens(x, y)) {
			insideAnyLens = true;
			//std::cout << delta << std::endl;
			l->ChangeClipDepth(delta*0.1, &matrix_mv.v[0].x, &matrix_pj.v[0].x);
		}
	}
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	return insideAnyLens;
}
