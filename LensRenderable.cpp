#include "LensRenderable.h"
#include "GlyphRenderable.h"
#include "Lens.h"
#include "glwidget.h"

void LensRenderable::init()
{
}

void LensRenderable::UpdateData() {
}

void LensRenderable::adjustOffset(){
	for (int i = 0; i < lenses.size(); i++) {
		Lens* l = lenses[i];

		if (l->type == LENS_TYPE::TYPE_CURVEB) {
			((CurveBLens*)l)->adjustOffset();

		}
	}

}; 

void LensRenderable::RefineLensBoundary(){
	for (int i = 0; i < lenses.size(); i++) {
		Lens* l = lenses[i];

		if (l->type == LENS_TYPE::TYPE_CURVEB) {
			((CurveBLens*)l)->RefineLensBoundary();

		}
	}

};

void LensRenderable::draw(float modelview[16], float projection[16])
{
	RecordMatrix(modelview, projection);

	int2 winSize = actor->GetWindowSize();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, winSize.x - 1, 0.0, winSize.y - 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);
	glLineWidth(4);
	glColor3f(1.0f, 0.2f, 0.2f);
	for (int i = 0; i < lenses.size(); i++) {
		Lens* l = lenses[i];

		if (l->type == LENS_TYPE::TYPE_POLYLINE) {
			glColor3f(0.2f, 1.0f, 0.2f);

			std::vector<float2> lensExtraRendering = ((PolyLineLens*)l)->GetExtraLensRendering();
			glBegin(GL_LINE_STRIP);
			for (auto v : lensExtraRendering)
				glVertex2f(v.x, v.y);
			glEnd();

			glColor3f(1.0f, 0.2f, 0.2f);

		}
		else if (l->type == LENS_TYPE::TYPE_CURVE) {
			glColor3f(0.2f, 1.0f, 0.2f);
			std::vector<float2> lensExtraRendering = l->GetExtraLensRendering();
			glBegin(GL_LINE_STRIP);
			for (auto v : lensExtraRendering)
				glVertex2f(v.x, v.y);
			glEnd();

			glColor3f(1.0f, 0.2f, 0.2f);
			std::vector<float2> lensContour = l->GetContour();
			glBegin(GL_LINE_LOOP);
			for (auto v : lensContour)
				glVertex2f(v.x, v.y);
			glEnd();

			glColor3f(0.2f, 0.8f, 0.8f);
			std::vector<float2> lensOuterContour = l->GetOuterContour();
			glBegin(GL_LINE_LOOP);
			for (auto v : lensOuterContour)
				glVertex2f(v.x, v.y);
			glEnd();
		}
		else if (l->type == LENS_TYPE::TYPE_CURVEB) {
			glLineWidth(1);
			glColor3f(0.2f, 1.0f, 0.2f);
			std::vector<float2> lensExtraRendering = l->GetExtraLensRendering();
			glPointSize(1.0);
			//glBegin(GL_POINTS);
			glBegin(GL_LINE_STRIP);
			for (auto v : lensExtraRendering)
				glVertex2f(v.x, v.y);
			glEnd();
			glLineWidth(4);
			glColor3f(0.9f, 0.9f, 0.2f);
			std::vector<float2> lensExtraRendering2 = ((CurveBLens *)l)->GetExtraLensRendering2();
			glPointSize(5.0);
			//glBegin(GL_POINTS);
			glBegin(GL_LINE_STRIP);
			for (auto v : lensExtraRendering2)
				glVertex2f(v.x, v.y);
			glEnd();

			glLineWidth(1);
			glColor3f(1.0f, 0.2f, 0.2f);
			std::vector<float2> lensContour = l->GetContour();
			glBegin(GL_LINE_LOOP);
			for (auto v : lensContour)
				glVertex2f(v.x, v.y);
			glEnd();


			vector<float2> pp = ((CurveBLens *)l)->posOffsetCtrlPoints;
			vector<float2> nn = ((CurveBLens *)l)->negOffsetCtrlPoints;
			
			vector<float2> pb = ((CurveBLens *)l)->posOffsetBezierPoints;
			vector<float2> nb = ((CurveBLens *)l)->negOffsetBezierPoints;
			vector<float2> subp = ((CurveBLens *)l)->subCtrlPointsPos;
			vector<float2> subn = ((CurveBLens *)l)->subCtrlPointsNeg;

			float2 center = make_float2(((CurveBLens *)l)->x, ((CurveBLens *)l)->y);
			for (int ii = 0; ii < pp.size(); ii++){
				pp[ii] = pp[ii] + center;
				pb[ii] = pb[ii] + center;
				subp[ii] = subp[ii] + center;
			}
			for (int ii = 0; ii < nn.size(); ii++){
				nn[ii] = nn[ii] + center;
				nb[ii] = nb[ii] + center;
				subn[ii] = subn[ii] + center;
			}
			glColor3f(0.2f, 0.8f, 0.8f);
			glPointSize(3.0);
			glBegin(GL_POINTS);
			for (auto v : pp)
				glVertex2f(v.x, v.y);
			for (auto v : nn)
				glVertex2f(v.x, v.y);
			glEnd();

			glColor3f(0.2f, 0.2f, 0.8f);
			glBegin(GL_LINES);
			for (int ii = 0; ii < pp.size(); ii++){
				glVertex2f(pb[ii].x, pb[ii].y);
				glVertex2f(subp[ii].x, subp[ii].y);
			}
			for (int ii = 0; ii < nn.size(); ii++){
				glVertex2f(nb[ii].x, nb[ii].y);
				glVertex2f(subn[ii].x, subn[ii].y);
			}
			glEnd();




			//glColor3f(0.2f, 0.8f, 0.8f);
			//std::vector<float2> lensOuterContour = l->GetOuterContour();
			//glBegin(GL_LINE_LOOP);
			//for (auto v : lensOuterContour)
			//	glVertex2f(v.x, v.y);
			//glEnd();
			glLineWidth(4);
		}
		else if (l->type == LENS_TYPE::TYPE_CIRCLE || l->type == LENS_TYPE::TYPE_LINE){
			std::vector<float2> lensContour = l->GetContour();
			glBegin(GL_LINE_LOOP);
			for (auto v : lensContour)
				glVertex2f(v.x, v.y);
			glEnd();

			glColor3f(0.2f, 0.8f, 0.8f);
			std::vector<float2> lensOuterContour = l->GetOuterContour();
			glBegin(GL_LINE_LOOP);
			for (auto v : lensOuterContour)
				glVertex2f(v.x, v.y);
			glEnd();
		}
	}

	glPopAttrib();
	//restore the original 3D coordinate system
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void LensRenderable::AddCircleLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new CircleLens(winSize.x * 0.5, winSize.y * 0.5, winSize.y * 0.2, actor->DataCenter());
	lenses.push_back(l);
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
}

void LensRenderable::AddLineLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new LineLens(winSize.x * 0.5, winSize.y * 0.5, winSize.y * 0.2, actor->DataCenter());
	lenses.push_back(l);
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
}

void LensRenderable::AddPolyLineLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new PolyLineLens(winSize.x * 0.5, winSize.y * 0.5, winSize.y * 0.05, actor->DataCenter());
	lenses.push_back(l);
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
	actor->SetInteractMode(INTERACT_MODE::MODIFYING_LENS);
}

void LensRenderable::AddCurveLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new CurveLens(winSize.x * 0.5, winSize.y * 0.5, winSize.y * 0.1, actor->DataCenter());
	lenses.push_back(l);
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
	actor->SetInteractMode(INTERACT_MODE::MODIFYING_LENS);
}

void LensRenderable::AddCurveBLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new CurveBLens(winSize.x * 0.5, winSize.y * 0.5, winSize.y * 0.1, actor->DataCenter());
	lenses.push_back(l);
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
	actor->SetInteractMode(INTERACT_MODE::MODIFYING_LENS);
}

void LensRenderable::mousePress(int x, int y, int modifier)
{
	if (INTERACT_MODE::MODIFYING_LENS == actor->GetInteractMode()) {
		for (int i = 0; i < lenses.size(); i++) {
			Lens* l = lenses[i];
			if (l->type == LENS_TYPE::TYPE_POLYLINE) {
				if (modifier == Qt::ControlModifier) {
					((PolyLineLens *)l)->AddCtrlPoint(x, y);
				}
				else{
					((PolyLineLens *)l)->FinishConstructing();
					actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				}
			}
			else if (l->type == LENS_TYPE::TYPE_CURVE) {
				if (modifier == Qt::ControlModifier) {
					((CurveLens *)l)->AddCtrlPoint(x, y);
				}
			}
			else if ( l->type == LENS_TYPE::TYPE_CURVEB) {
				if (modifier == Qt::ControlModifier) {
					((CurveBLens *)l)->AddCtrlPoint(x, y);
				}
			}
		}
	}
	else {
		for (int i = 0; i < lenses.size(); i++) {
			Lens* l = lenses[i];
			if (l->PointInsideLens(x, y)) {
				//workingOnLens = true;
				actor->SetInteractMode(INTERACT_MODE::LENS);
				pickedLens = i;
				lastPt = make_int2(x, y);
			}
		}
	}
	//return insideAnyLens;
}

void LensRenderable::mouseRelease(int x, int y, int modifier)
{
	if (INTERACT_MODE::MODIFYING_LENS == actor->GetInteractMode()) {
		Lens* l = lenses[lenses.size() - 1];
		if (l->type == LENS_TYPE::TYPE_CURVE) {
			((CurveLens *)l)->FinishConstructing();
			actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
		}
		else if (l->type == LENS_TYPE::TYPE_CURVEB) {
			((CurveBLens *)l)->FinishConstructing();
			actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
		}
		else{
			//polyline lens
		}
	}
	else {
		actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
	}

	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
}

void LensRenderable::mouseMove(int x, int y, int modifier)
{
	if (INTERACT_MODE::MODIFYING_LENS == actor->GetInteractMode()) {
		Lens* l = lenses[lenses.size() - 1];
		if (l->type == LENS_TYPE::TYPE_CURVE || l->type == LENS_TYPE::TYPE_CURVEB){
			((CurveLens *)l)->AddCtrlPoint(x, y);
		}
	}
	else{

		if (actor->GetInteractMode() == INTERACT_MODE::LENS) {
			lenses[pickedLens]->x += (x - lastPt.x);
			lenses[pickedLens]->y += (y - lastPt.y);
		}
		((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
		lastPt = make_int2(x, y);
	}
}

bool LensRenderable::MouseWheel(int x, int y, int delta)
{
	bool insideAnyLens = false;
	for (int i = 0; i < lenses.size(); i++) {
		Lens* l = lenses[i];
		if (l->PointInsideLens(x, y)) {
			insideAnyLens = true;
			//std::cout << delta << std::endl;
			l->ChangeClipDepth(delta*0.05, &matrix_mv.v[0].x, &matrix_pj.v[0].x);
		}
	}
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	return insideAnyLens;
}

void LensRenderable::SlotFocusSizeChanged(int v)
{
	if (lenses.size() > 0){
		lenses.back()->SetFocusRatio((10 - v) * 0.1 * 0.8 + 0.2);
		Lens *l = lenses.back();
		if (l->GetType() == LENS_TYPE::TYPE_CURVE)
		{
			((CurveLens *)l)->UpdateTransferredData();
		}
	}
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
}


void LensRenderable::SlotSideSizeChanged(int v)// { displace - (10 - v) * 0.1; }
{
	if (lenses.size() > 0){
		lenses.back()->SetSideSize(v * 0.1);
	}
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
}

void LensRenderable::SlotDelLens()
{
	if (lenses.size() > 0){
		lenses.pop_back();
	}
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
}