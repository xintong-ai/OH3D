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

	if (1){

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

			if (l->type == LENS_TYPE::TYPE_CURVEB) {
				//glLineWidth(1);
				glColor3f(0.2f, 1.0f, 0.2f);
				std::vector<float2> lensExtraRendering = ((CurveBLens *)l)->GetCtrlPointsForRendering(modelview, projection, winSize.x, winSize.y);
				glPointSize(5.0);
				glBegin(GL_POINTS);
				for (int i = 0; i < lensExtraRendering.size(); i++){
					glColor3f(1.0f - i % 4 / 3.0, 0.0f, i % 4 / 3.0);

					float2 v = lensExtraRendering[i];
					glVertex2f(v.x, v.y);

				}

				//for (auto v : lensExtraRendering)
				//	glVertex2f(v.x, v.y);
				glEnd();

				//glLineWidth(4);
				glColor3f(0.9f, 0.9f, 0.2f);
				std::vector<float2> lensExtraRendering2 = ((CurveBLens *)l)->GetCenterLineForRendering(modelview, projection, winSize.x, winSize.y);
				glPointSize(5.0);
				//glBegin(GL_POINTS);
				glBegin(GL_LINE_STRIP);
				for (auto v : lensExtraRendering2)
					glVertex2f(v.x, v.y);
				glEnd();

				//glLineWidth(1);
				glColor3f(1.0f, 0.2f, 0.2f);
				std::vector<float2> lensContour = l->GetContour(modelview, projection, winSize.x, winSize.y);
				glBegin(GL_LINE_LOOP);
				for (auto v : lensContour)
					glVertex2f(v.x, v.y);
				glEnd();


				//vector<float2> pp = ((CurveBLens *)l)->posOffsetCtrlPoints;
				//vector<float2> nn = ((CurveBLens *)l)->negOffsetCtrlPoints;
				//
				//vector<float2> pb = ((CurveBLens *)l)->posOffsetBezierPoints;
				//vector<float2> nb = ((CurveBLens *)l)->negOffsetBezierPoints;
				//vector<float2> subp = ((CurveBLens *)l)->subCtrlPointsPos;
				//vector<float2> subn = ((CurveBLens *)l)->subCtrlPointsNeg;

				//float2 center = make_float2(((CurveBLens *)l)->x, ((CurveBLens *)l)->y);
				//for (int ii = 0; ii < pp.size(); ii++){
				//	pp[ii] = pp[ii] + center;
				//	pb[ii] = pb[ii] + center;
				//	subp[ii] = subp[ii] + center;
				//}
				//for (int ii = 0; ii < nn.size(); ii++){
				//	nn[ii] = nn[ii] + center;
				//	nb[ii] = nb[ii] + center;
				//	subn[ii] = subn[ii] + center;
				//}
				//glColor3f(0.2f, 0.8f, 0.8f);
				//glPointSize(3.0);
				//glBegin(GL_POINTS);
				//for (auto v : pp)
				//	glVertex2f(v.x, v.y);
				//for (auto v : nn)
				//	glVertex2f(v.x, v.y);
				//glEnd();

				//glColor3f(0.2f, 0.2f, 0.8f);
				//glBegin(GL_LINES);
				//for (int ii = 0; ii < pp.size(); ii++){
				//	glVertex2f(pb[ii].x, pb[ii].y);
				//	glVertex2f(subp[ii].x, subp[ii].y);
				//}
				//for (int ii = 0; ii < nn.size(); ii++){
				//	glVertex2f(nb[ii].x, nb[ii].y);
				//	glVertex2f(subn[ii].x, subn[ii].y);
				//}
				//glEnd();




				glColor3f(0.2f, 0.8f, 0.8f);
				std::vector<float2> lensOuterContour = l->GetOuterContour(modelview, projection, winSize.x, winSize.y);
				glBegin(GL_LINE_LOOP);
				for (auto v : lensOuterContour)
					glVertex2f(v.x, v.y);
				glEnd();
				//glLineWidth(4);
			}
			else if (l->type == LENS_TYPE::TYPE_CIRCLE || l->type == LENS_TYPE::TYPE_LINE){
				std::vector<float2> lensContour = l->GetContour(modelview, projection, winSize.x, winSize.y);
				glBegin(GL_LINE_LOOP);
				for (auto v : lensContour)
					glVertex2f(v.x, v.y);
				glEnd();

				glColor3f(0.2f, 0.8f, 0.8f);
				std::vector<float2> lensOuterContour = l->GetOuterContour(modelview, projection, winSize.x, winSize.y);
				glBegin(GL_LINE_LOOP);
				for (auto v : lensOuterContour)
					glVertex2f(v.x, v.y);
				glEnd();
			}
			else if (l->type == LENS_TYPE::TYPE_LINEB){
				std::vector<float2> lensContour = l->GetContour(modelview, projection, winSize.x, winSize.y);
				glBegin(GL_LINE_LOOP);
				for (auto v : lensContour)
					glVertex2f(v.x, v.y);
				glEnd();

				glColor3f(0.2f, 0.8f, 0.8f);
				std::vector<float2> lensOuterContour = l->GetOuterContour(modelview, projection, winSize.x, winSize.y);
				glBegin(GL_LINE_LOOP);
				for (auto v : lensOuterContour)
					glVertex2f(v.x, v.y);
				glEnd();

				std::vector<float2> ctrlPoints = l->GetCtrlPointsForRendering(modelview, projection, winSize.x, winSize.y);
				glColor3f(0.8f, 0.8f, 0.2f);
				if (((LineBLens*)l)->isConstructing){
					glBegin(GL_LINES);
					for (auto v : ctrlPoints)
						glVertex2f(v.x, v.y);
					glEnd();
				}
				else{
					glPointSize(10.0);
					glBegin(GL_POINTS);
					for (auto v : ctrlPoints)
						glVertex2f(v.x, v.y);
					glEnd();
				}
			}
		}

		glPopAttrib();
		//restore the original 3D coordinate system
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
	}
	else{//to draw 3D contour
		glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);
		glLineWidth(4);
		glColor3f(1.0f, 0.2f, 0.2f);

		for (int i = 0; i < lenses.size(); i++) {
			Lens* l = lenses[i];

			if (l->type == LENS_TYPE::TYPE_CIRCLE || l->type == LENS_TYPE::TYPE_LINE){
				std::vector<std::vector<float3>> lensContour = ((CircleLens*)l)->Get3DContour();
				
				for (int i = 0; i <lensContour.size()-1; i++){
					glBegin(GL_LINE_LOOP);
					for (auto v : lensContour[i]){
						glVertex3f(v.x, v.y, v.z);
					}
					glEnd();
				}

				glBegin(GL_LINES);
				for (auto v : lensContour[lensContour.size() - 1]){
					glVertex3f(v.x, v.y, v.z);
				}
				glEnd();

			}
		}
		glPopAttrib();

	}
}
void LensRenderable::AddCircleLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new CircleLens(winSize.y * 0.2, actor->DataCenter());
	lenses.push_back(l);
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
}

void LensRenderable::AddLineLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new LineLens(winSize.y * 0.2, actor->DataCenter());
	lenses.push_back(l);
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
}

void LensRenderable::AddLineBLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new LineBLens(actor->DataCenter());
	lenses.push_back(l);
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
	actor->SetInteractMode(INTERACT_MODE::MODIFYING_LENS);

}

void LensRenderable::AddCurveBLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new CurveBLens(winSize.y * 0.1, actor->DataCenter());
	lenses.push_back(l);
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
	actor->SetInteractMode(INTERACT_MODE::MODIFYING_LENS);
}

void LensRenderable::mousePress(int x, int y, int modifier)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	if (INTERACT_MODE::MODIFYING_LENS == actor->GetInteractMode()) {
		for (int i = 0; i < lenses.size(); i++) {
			Lens* l = lenses[i];
			if (l->type == LENS_TYPE::TYPE_CURVEB) {
				((CurveBLens *)l)->AddCtrlPoint(x, y);
			}
			else if (l->type == LENS_TYPE::TYPE_LINEB) {
				((LineBLens *)l)->ctrlPoint1Abs = make_float2(x, y);
				((LineBLens *)l)->ctrlPoint2Abs = make_float2(x, y);
			}
		}
	}
	else {
		for (int i = 0; i < lenses.size(); i++) {
			Lens* l = lenses[i];
			if (l->PointInsideLens(x, y, modelview, projection, winSize.x, winSize.y)) {
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
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	if (INTERACT_MODE::MODIFYING_LENS == actor->GetInteractMode()) {
		Lens* l = lenses[lenses.size() - 1];
		if (l->type == LENS_TYPE::TYPE_CURVEB) {
			((CurveBLens *)l)->FinishConstructing(modelview, projection, winSize.x, winSize.y);
			actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
		}
		else if (l->type == LENS_TYPE::TYPE_LINEB) {
			((LineBLens*)l)->FinishConstructing(modelview, projection, winSize.x, winSize.y);
			actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
		}

	}
	else {
		if (actor->GetInteractMode() == INTERACT_MODE::LENS && isSnapToGlyph && modifier != Qt::AltModifier){
			GlyphRenderable* glyphRenderable = (GlyphRenderable*)actor->GetRenderable("glyph");
			Lens* l = lenses[lenses.size() - 1];
			float3 center = make_float3(l->GetCenter());
			float3 snapPos = glyphRenderable->findClosetGlyph(center);
			l->SetCenter(snapPos);
		}
		else if (actor->GetInteractMode() == INTERACT_MODE::LENS && isSnapToFeature && modifier != Qt::ShiftModifier){
			GlyphRenderable* glyphRenderable = (GlyphRenderable*)actor->GetRenderable("glyph");
			Lens* l = lenses[lenses.size() - 1];
			float3 center = make_float3(l->GetCenter());
			float3 snapPos;
			if (glyphRenderable->findClosetFeature(center, snapPos))
				l->SetCenter(snapPos);
		}
		actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
	}

	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
}

void LensRenderable::mouseMove(int x, int y, int modifier)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	if (INTERACT_MODE::MODIFYING_LENS == actor->GetInteractMode()) {
		Lens* l = lenses[lenses.size() - 1];
		if (l->type == LENS_TYPE::TYPE_CURVEB){
			((CurveBLens *)l)->AddCtrlPoint(x, y);
		}
		else if (l->type == LENS_TYPE::TYPE_LINEB){
			((LineBLens *)l)->ctrlPoint2Abs = make_float2(x, y);
			((LineBLens*)l)->UpdateInfo(modelview, projection, winSize.x, winSize.y);
			((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
		}
	}
	else{
		if (actor->GetInteractMode() == INTERACT_MODE::LENS) {
			//lenses[pickedLens]->x += (x - lastPt.x);
			//lenses[pickedLens]->y += (y - lastPt.y);
			float2 center = lenses[pickedLens]->GetCenterScreenPos(modelview, projection, winSize.x, winSize.y);
			lenses[pickedLens]->UpdateCenterByScreenPos(
				center.x + (x - lastPt.x), center.y + (y - lastPt.y)
				, modelview, projection, winSize.x, winSize.y);
			if (isSnapToGlyph && modifier != Qt::AltModifier){
				GlyphRenderable* glyphRenderable = (GlyphRenderable*)actor->GetRenderable("glyph");
				glyphRenderable->findClosetGlyph(make_float3(lenses[pickedLens]->GetCenter()));
			}
			else if (isSnapToFeature && modifier != Qt::ShiftModifier){
				GlyphRenderable* glyphRenderable = (GlyphRenderable*)actor->GetRenderable("glyph");
				float3 snapPos;
				glyphRenderable->findClosetFeature(make_float3(lenses[pickedLens]->GetCenter()), snapPos);
			}
		}
		((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
		lastPt = make_int2(x, y);
	}
}

bool LensRenderable::MouseWheel(int x, int y, int modifier, int delta)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	bool insideAnyLens = false;
	for (int i = 0; i < lenses.size(); i++) {
		Lens* l = lenses[i];
		if (l->PointInsideLens(x, y, modelview, projection, winSize.x, winSize.y)) {
			insideAnyLens = true;
			//std::cout << delta << std::endl;
			l->ChangeClipDepth(delta*0.05, &matrix_mv.v[0].x, &matrix_pj.v[0].x);
			//if (isSnapToGlyph && modifier != Qt::AltModifier){
			//	GlyphRenderable* glyphRenderable = (GlyphRenderable*)actor->GetRenderable("glyph");
			//	glyphRenderable->findClosetGlyph(make_float3(l->GetCenter()));
			//}
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
		//if (l->GetType() == LENS_TYPE::TYPE_CURVE)
		//{
		//	((CurveLens *)l)->UpdateTransferredData();
		//}
	}
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
}


void LensRenderable::SlotSideSizeChanged(int v)// { displace - (10 - v) * 0.1; }
{
	//if (lenses.size() > 0){
	//	lenses.back()->SetSideSize(v * 0.1);
	//}
	//((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	//actor->UpdateGL();
}

void LensRenderable::SlotDelLens()
{
	if (lenses.size() > 0){
		lenses.pop_back();
	}
	((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
	actor->UpdateGL();
}

void LensRenderable::SlotLensCenterChanged(float3 p)
{
	if (lenses.size() > 0){
		int2 winSize = actor->GetWindowSize();
		GLfloat modelview[16];
		GLfloat projection[16];
		actor->GetModelview(modelview);
		actor->GetProjection(projection);
		//lenses.back()->c = p;
		float3 pScreen;
		pScreen.x = (p.x + 117.5) / 235.0 * winSize.x;
		pScreen.y = (p.y - 82.5) / 235.0 * winSize.y;
		const float aa = 0.02f;
		float2 depthRange;
		actor->GetDepthRange(depthRange);
		//pScreen.z = clamp((1.0f - aa * , 0.0f, 1.0f);
		std::cout << "bb:" << clamp((1.0 - (p.z + 73.5f) / 147.0f), 0.0f, 1.0f) << std::endl;
		pScreen.z = depthRange.x + (depthRange.y - depthRange.x) * clamp((1.0 - (p.z + 73.5f) / 147.0f), 0.0f, 1.0f);
		std::cout << "depth:" << pScreen.z << std::endl;
		lenses.back()->SetClipDepth(pScreen.z, &matrix_mv.v[0].x, &matrix_pj.v[0].x);
		//pScreen.z = clamp(aa *(1 - (p.y + 73.5) / 147), 0, 1);
		//lenses.back()->c.z = pScreen.z;
		lenses.back()->UpdateCenterByScreenPos(pScreen.x, pScreen.y, modelview, projection, winSize.x, winSize.y);
		//interaction box
		//https://developer.leapmotion.com/documentation/csharp/devguide/Leap_Coordinate_Mapping.html
		((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
		actor->UpdateGL();
	}
}
