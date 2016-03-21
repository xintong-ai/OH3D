#include "LensRenderable.h"
#include "GlyphRenderable.h"
#include "Lens.h"
#include "glwidget.h"

class SolidSphere
{
protected:
	std::vector<GLfloat> vertices;
	std::vector<GLfloat> normals;
	std::vector<GLfloat> texcoords;
	std::vector<GLushort> indices;

public:
	SolidSphere(float radius, unsigned int rings, unsigned int sectors)
	{
		float const R = 1. / (float)(rings - 1);
		float const S = 1. / (float)(sectors - 1);
		int r, s;

		vertices.resize(rings * sectors * 3);
		normals.resize(rings * sectors * 3);
		texcoords.resize(rings * sectors * 2);
		std::vector<GLfloat>::iterator v = vertices.begin();
		std::vector<GLfloat>::iterator n = normals.begin();
		std::vector<GLfloat>::iterator t = texcoords.begin();
		for (r = 0; r < rings; r++) for (s = 0; s < sectors; s++) {
			float const y = sin(-M_PI_2 + M_PI * r * R);
			float const x = cos(2 * M_PI * s * S) * sin(M_PI * r * R);
			float const z = sin(2 * M_PI * s * S) * sin(M_PI * r * R);

			*t++ = s*S;
			*t++ = r*R;

			*v++ = x * radius;
			*v++ = y * radius;
			*v++ = z * radius;

			*n++ = x;
			*n++ = y;
			*n++ = z;
		}

		indices.resize(rings * sectors * 4);
		std::vector<GLushort>::iterator i = indices.begin();
		for (r = 0; r < rings - 1; r++) for (s = 0; s < sectors - 1; s++) {
			*i++ = r * sectors + s;
			*i++ = r * sectors + (s + 1);
			*i++ = (r + 1) * sectors + (s + 1);
			*i++ = (r + 1) * sectors + s;
		}
	}

	void draw(GLfloat x, GLfloat y, GLfloat z)
	{
		glColor3f(1.0f, 1.0f, 1.0f);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslatef(x, y, z);
		glScalef(0.2, 0.2, 0.2);

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);

		glVertexPointer(3, GL_FLOAT, 0, &vertices[0]);
		glNormalPointer(GL_FLOAT, 0, &normals[0]);
		glTexCoordPointer(2, GL_FLOAT, 0, &texcoords[0]);
		glDrawElements(GL_QUADS, indices.size(), GL_UNSIGNED_SHORT, &indices[0]);
		glPopMatrix();
	}
};

LensRenderable::LensRenderable()
{
	lensCenterSphere = new SolidSphere(0.1, 12, 24);

}
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

	for (int i = 0; i < lenses.size(); i++) {
		Lens* l = lenses[i];
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glLoadMatrixf(projection);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glLoadMatrixf(modelview);

		lensCenterSphere->draw(l->c.x, l->c.y, l->c.z);

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();

	}

	if (DEFORM_MODEL::SCREEN_SPACE == actor->GetDeformModel()){

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
		for (int i = 0; i < lenses.size(); i++) {
			Lens* l = lenses[i];
			std::vector<float2> lensContour = l->GetContour(modelview, projection, winSize.x, winSize.y);
			glColor3f(0.39f, 0.89f, 0.26f);
			glBegin(GL_LINE_LOOP);
			for (auto v : lensContour)
				glVertex2f(v.x, v.y);
			glEnd();

			glColor3f(0.82f, 0.31f, 0.67f);
			std::vector<float2> lensOuterContour = l->GetOuterContour(modelview, projection, winSize.x, winSize.y);
			glBegin(GL_LINE_LOOP);
			for (auto v : lensOuterContour)
				glVertex2f(v.x, v.y);
			glEnd();
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
			}

			if (l->type == LENS_TYPE::TYPE_LINEB){
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
	else if (DEFORM_MODEL::OBJECT_SPACE == actor->GetDeformModel()){
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

float3 LensRenderable::GetBackLensCenter()
{
	float3 ret = make_float3(0,0,5);
	if (lenses.size() > 0)
		ret = lenses.back()->c;
	return ret;
}

bool LensRenderable::InsideALens(int x, int y)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	bool ret = false;
	for (int i = 0; i < lenses.size(); i++) {
		Lens* l = lenses[i];
		if (l->PointInsideLens(x, y, modelview, projection, winSize.x, winSize.y)) {
			ret = true;
			break;
		}
	}
	return ret;
}


void LensRenderable::mousePress(int x, int y, int modifier)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	switch (actor->GetInteractMode())
	{
	case INTERACT_MODE::MODIFYING_LENS:
	{
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
		break;
	}
	case INTERACT_MODE::TRANSFORMATION:
	{
		for (int i = 0; i < lenses.size(); i++) {
			Lens* l = lenses[i];
			if (l->PointOnLensCenter(x, y, modelview, projection, winSize.x, winSize.y)) {
				//workingOnLens = true;
				actor->SetInteractMode(INTERACT_MODE::MOVE_LENS);
				pickedLens = i;
				break;
			}
			else if (l->PointOnInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
				actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE);
				pickedLens = i;
				break;
			}
			else if (l->PointOnOuterBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
				actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE);
				pickedLens = i;
				break;
			}
			else if (l->PointInsideLens(x, y, modelview, projection, winSize.x, winSize.y)) {
				actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_DEPTH);
				pickedLens = i;
				break;
			}
		}
		break;
	}

	}
	//return insideAnyLens;
	lastPt = make_int2(x, y);
	std::cout << lastPt.x << ", " << lastPt.y << std::endl;
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
		if (actor->GetInteractMode() == INTERACT_MODE::MOVE_LENS && isSnapToGlyph && modifier != Qt::AltModifier){
			GlyphRenderable* glyphRenderable = (GlyphRenderable*)actor->GetRenderable("glyph");
			Lens* l = lenses[lenses.size() - 1];
			float3 center = make_float3(l->GetCenter());
			float3 snapPos = glyphRenderable->findClosetGlyph(center);
			l->SetCenter(snapPos);
		}
		else if (actor->GetInteractMode() == INTERACT_MODE::MOVE_LENS && isSnapToFeature && modifier != Qt::ShiftModifier){
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
	switch (actor->GetInteractMode())
	{
	case INTERACT_MODE::MODIFYING_LENS:
	{
		Lens* l = lenses[lenses.size() - 1];
		if (l->type == LENS_TYPE::TYPE_CURVEB){
			((CurveBLens *)l)->AddCtrlPoint(x, y);
		}
		else if (l->type == LENS_TYPE::TYPE_LINEB){
			((LineBLens *)l)->ctrlPoint2Abs = make_float2(x, y);
			((LineBLens*)l)->UpdateInfo(modelview, projection, winSize.x, winSize.y);
			((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
		}
		break;
	}
	case INTERACT_MODE::MOVE_LENS:
	{
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
		((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
		break;
	}
	case INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE:
	{
		lenses[pickedLens]->ChangeLensSize(x, y, modelview, projection, winSize.x, winSize.y);
		break;
	}
	case INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE:
	{
		lenses[pickedLens]->ChangefocusRatio(x, y, modelview, projection, winSize.x, winSize.y);
		break;
	}
	}
	lastPt = make_int2(x, y);
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

void LensRenderable::PinchScaleFactorChanged(float x, float y, float totalScaleFactor)
{
	//GLfloat modelview[16];
	//GLfloat projection[16];
	//actor->GetModelview(modelview);
	//actor->GetProjection(projection);
	//int2 winSize = actor->GetWindowSize();
	//pickedLens = -1;
	//for (int i = 0; i < lenses.size(); i++) {
	//	Lens* l = lenses[i];
	//	if (l->PointInsideLens(x, y, modelview, projection, winSize.x, winSize.y)) {
	//		pickedLens = i;
	//		break;
	//	}
	//}
	//if (pickedLens > -1){
	if (INTERACT_MODE::MODIFY_LENS_DEPTH == actor->GetInteractMode()){
		//actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_DEPTH);
		//std::cout << "totalScaleFactor:" << totalScaleFactor << std::endl;
		float scaleFactor = totalScaleFactor > 1 ? 1 : -1;
		lenses[pickedLens]->ChangeClipDepth(scaleFactor, &matrix_mv.v[0].x, &matrix_pj.v[0].x);
		((GlyphRenderable*)actor->GetRenderable("glyph"))->RecomputeTarget();
		actor->UpdateGL();
	}
	//}
	//else {
	//	actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
	//}
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

inline float3 GetNormalizedLeapPos(float3 p)
{
	float3 leapPos;
	leapPos.x = clamp((p.x + 117.5) / 235.0, 0.0f, 1.0f);
	leapPos.y = clamp((p.y - 82.5) / 235.0, 0.0f, 1.0f);
	leapPos.z = clamp((p.z + 73.5f) / 147.0f, 0.0f, 1.0f);
	return leapPos;
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
		float3 leapPos = GetNormalizedLeapPos(p);
		const float aa = 0.02f;
		float2 depthRange;
		actor->GetDepthRange(depthRange);
		//pScreen.z = clamp((1.0f - aa * , 0.0f, 1.0f);
		//std::cout << "bb:" << clamp((1.0 - (p.z + 73.5f) / 147.0f), 0.0f, 1.0f) << std::endl;
		
		//std::cout << "leapPos:" << leapPos.x << "," << leapPos.y << "," << leapPos.z << std::endl;
		bool usingVR = true;
		if (usingVR){
			pScreen.x = (1.0 - leapPos.x) * winSize.x;
			pScreen.y = clamp((1.0 - leapPos.z) * 2, 0.0f, 1.0f) * winSize.y;
			pScreen.z = depthRange.x + (depthRange.y - depthRange.x) * leapPos.y;
		}
		else{
			pScreen.x = leapPos.x * winSize.x;
			pScreen.y = leapPos.y * winSize.y;
			pScreen.z = depthRange.x + (depthRange.y - depthRange.x) * (1.0 - leapPos.z);
		}
		//std::cout << "depth:" << pScreen.z << std::endl;
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

/*
//code from chengli for object space drawing
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
*/
