#include "LensRenderable.h"
#include "DeformGlyphRenderable.h"
#include "Lens.h"
#include "DeformGLWidget.h"
#include "GLSphere.h"
#include "PolyRenderable.h"

// this class is used to draw the lens center
class SolidSphere
{
protected:
	std::vector<GLfloat> vertices;
	std::vector<GLfloat> normals;
	std::vector<GLfloat> texcoords;
	std::vector<GLushort> indices;

	std::vector<float3> grid;

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

			indices.resize(rings * sectors * 4);
			std::vector<GLushort>::iterator i = indices.begin();
			for (r = 0; r < rings - 1; r++) for (s = 0; s < sectors - 1; s++) {
				*i++ = r * sectors + s;
				*i++ = r * sectors + (s + 1);
				*i++ = (r + 1) * sectors + (s + 1);
				*i++ = (r + 1) * sectors + s;
			}
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

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);

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
	
	if (!visible)
		return;

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

		//lensCenterSphere->draw(l->c.x, l->c.y, l->c.z);
		glEnableClientState(GL_VERTEX_ARRAY);

		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		GLSphere sphere(1.0,2);
		glColor4f(1.0f, 1.0f, 1.0f, 0.9f);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glTranslatef(l->c.x, l->c.y, l->c.z);
		glScalef(0.1, 0.1, 0.1);
		glVertexPointer(3, GL_FLOAT, 0, sphere.GetVerts());
		glDrawArrays(GL_QUADS, 0, sphere.GetNumVerts());
		glPopMatrix();
		glPolygonMode(GL_FRONT_AND_BACK ,GL_FILL);

		glDisableClientState(GL_VERTEX_ARRAY);

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();

	}

	//temporarily changed for line lens object space debugging
	if (DEFORM_MODEL::SCREEN_SPACE == ((DeformGLWidget*)actor)->GetDeformModel()){
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
				glLineWidth(2);
				glColor3f(0.9f, 0.9f, 0.2f);
				std::vector<float2> lensExtraRendering2 = ((CurveBLens *)l)->GetCenterLineForRendering(modelview, projection, winSize.x, winSize.y);
				//glBegin(GL_POINTS);
				glBegin(GL_LINE_STRIP);
				for (auto v : lensExtraRendering2)
					glVertex2f(v.x, v.y);
				glEnd();

				glColor3f(0.39f, 0.89f, 0.26f);
				std::vector<float2> lensContour = l->GetContour(modelview, projection, winSize.x, winSize.y);
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


				std::vector<float2> pp = ((CurveBLens *)l)->posOffsetCtrlPoints;
				std::vector<float2> nn = ((CurveBLens *)l)->negOffsetCtrlPoints;
				//
				//vector<float2> pb = ((CurveBLens *)l)->posOffsetBezierPoints;
				//vector<float2> nb = ((CurveBLens *)l)->negOffsetBezierPoints;
				std::vector<float2> subp = ((CurveBLens *)l)->subCtrlPointsPos;
				//vector<float2> subn = ((CurveBLens *)l)->subCtrlPointsNeg;

				float2 center = ((CurveBLens *)l)->GetCenterScreenPos(modelview, projection, winSize.x, winSize.y);
				for (int ii = 0; ii < pp.size(); ii++){
					pp[ii] = pp[ii] + center;
					//pb[ii] = pb[ii] + center;
					subp[ii] = subp[ii] + center;
				}
				for (int ii = 0; ii < nn.size(); ii++){
					nn[ii] = nn[ii] + center;
					//nb[ii] = nb[ii] + center;
					//subn[ii] = subn[ii] + center;
				}

				glPointSize(5.0);
				if (pp.size() > nn.size())
				{
					glColor3f(0.8f, 0.0f, 0.8f);
					glBegin(GL_POINTS);
					for (int i = 0; i < pp.size(); i+=2){
						float2 v = pp[i];
						glVertex2f(v.x, v.y);
					}
					for (int i = 0; i < nn.size(); i++){
						float2 v = nn[i];
						glVertex2f(v.x, v.y);
					}
					

					glColor3f(0.0, 0.0f, 1.0);
					for (int i = 0; i < subp.size(); i+=2){
						float2 v = subp[i];
						glVertex2f(v.x, v.y);
					}

					glEnd();
				}

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
				////draw the control points of the center
				//glColor3f(0.2f, 1.0f, 0.2f);
				//std::vector<float2> lensExtraRendering = ((CurveBLens *)l)->GetCtrlPointsForRendering(modelview, projection, winSize.x, winSize.y);	
				//glBegin(GL_POINTS);
				//glColor3f(0.0, 0.0f, 1.0);
				//for (int i = 0; i < lensExtraRendering.size(); i++){
				//	//glColor3f(1.0f - i % 4 / 3.0, 0.0f, i % 4 / 3.0);
				//	float2 v = lensExtraRendering[i];
				//	glVertex2f(v.x, v.y);
				//}
				//glEnd();
			}
		}

		glPopAttrib();
		//restore the original 3D coordinate system
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();

		
	}
	else if (DEFORM_MODEL::OBJECT_SPACE == ((DeformGLWidget*)actor)->GetDeformModel()){
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
			if (l->type == LENS_TYPE::TYPE_CIRCLE){ //same with screen space
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
			}
		}

		glPopAttrib();
		//restore the original 3D coordinate system
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();


		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glLoadMatrixf(projection);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glLoadMatrixf(modelview);

		glLineWidth(2);

		for (int i = 0; i < lenses.size(); i++) {
			Lens* l = lenses[i];
			//if (l->type == LENS_TYPE::TYPE_LINE){
			//	glColor3f(1 - 0.39f, 1 - 0.89f, 1 - 0.26f);
			//	std::vector<float3> lensContourGlobal = ((LineLens3D*)l)->GetContourGlobal(modelview, projection, winSize.x, winSize.y);
			//	glBegin(GL_LINE_LOOP);
			//	for (auto v : lensContourGlobal)
			//		glVertex3f(v.x, v.y, v.z);
			//	glEnd();

			//	glColor3f(1 - 0.82f, 1 - 0.31f, 1 - 0.67f);
			//	std::vector<float3> lensOuterContourGlobal = ((LineLens3D*)l)->GetOuterContourGlobal(modelview, projection, winSize.x, winSize.y);
			//	glBegin(GL_LINE_LOOP);
			//	for (auto v : lensOuterContourGlobal)
			//		glVertex3f(v.x, v.y, v.z);
			//	glEnd();
			//}
			if (l->type == LENS_TYPE::TYPE_LINE && !l->isConstructing){
				glColor3f(0.39f, 0.89f, 0.26f);
				std::vector<float3> PointsForContourBack = ((LineLens3D*)l)->PointsForContourBack;
				std::vector<float3> PointsForContourFront = ((LineLens3D*)l)->PointsForContourFront;;
				glBegin(GL_LINE_LOOP);
				for (auto v : PointsForContourBack)
					glVertex3f(v.x, v.y, v.z);
				glEnd();
				glBegin(GL_LINE_LOOP);
				for (auto v : PointsForContourFront)
					glVertex3f(v.x, v.y, v.z);
				glEnd();
				glBegin(GL_LINES);
				for (int i = 0; i < 4; i++){
					float3 v = PointsForContourBack[i], v2 = PointsForContourFront[i];
					glVertex3f(v.x, v.y, v.z);
					glVertex3f(v2.x, v2.y, v2.z);
				}
				glEnd();


				glColor3f(0.82f, 0.31f, 0.67f);
				std::vector<float3> PointsForContourOuterBack = ((LineLens3D*)l)->PointsForOuterContourBack;
				std::vector<float3> PointsForContourOuterFront = ((LineLens3D*)l)->PointsForOuterContourFront;;
				glBegin(GL_LINE_LOOP);
				for (auto v : PointsForContourOuterBack)
					glVertex3f(v.x, v.y, v.z);
				glEnd();
				glBegin(GL_LINE_LOOP);
				for (auto v : PointsForContourOuterFront)
					glVertex3f(v.x, v.y, v.z);
				glEnd();
				glBegin(GL_LINES);
				for (int i = 0; i < 4; i++){
					float3 v = PointsForContourOuterBack[i], v2 = PointsForContourOuterFront[i];
					glVertex3f(v.x, v.y, v.z);
					glVertex3f(v2.x, v2.y, v2.z);
				}
				glEnd();
			}
		}

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
	}
	
	bool draw3DContour = false;
	if (draw3DContour){
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
		glColor3f(1.0f, 0.2f, 0.2f);


		QMatrix4x4 q_modelview = QMatrix4x4(modelview);
		q_modelview = q_modelview.transposed();
		QMatrix4x4 q_invVP = q_modelview.inverted();
		QVector4D q_eye4 = q_invVP.map(QVector4D(0, 0, 0, 1));
		float3 eyeWorld;
		eyeWorld.x = q_eye4[0];
		eyeWorld.y = q_eye4[1];
		eyeWorld.z = q_eye4[2];

		for (int i = 0; i < lenses.size(); i++) {
			Lens* l = lenses[i];

			if (l->type == LENS_TYPE::TYPE_CIRCLE){
				std::vector<std::vector<float3>> lensContour = ((CircleLens*)l)->Get3DContour(eyeWorld, true);

				for (int i = 0; i < lensContour.size() - 1; i++){
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

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
		glPopAttrib();
	}
	
	if (draw3DContour)
	{
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


		QMatrix4x4 q_modelview = QMatrix4x4(modelview);
		q_modelview = q_modelview.transposed();
		QMatrix4x4 q_invVP = q_modelview.inverted();
		QVector4D q_eye4 = q_invVP.map(QVector4D(0, 0, 0, 1));
		float3 eyeWorld;
		eyeWorld.x = q_eye4[0];
		eyeWorld.y = q_eye4[1];
		eyeWorld.z = q_eye4[2];

		for (int i = 0; i < lenses.size(); i++) {
			Lens* l = lenses[i];

			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glPointSize(25.0);
			glEnable(GL_POINT_SMOOTH);

			glColor4f(0.5f, 0.3f, 0.8f, 0.9f);
			glBegin(GL_POINTS);

			//glVertex3f(l->c.x, l->c.y, l->c.z);

			glEnd();
			glDisable(GL_POINT_SMOOTH);


			if (l->type == LENS_TYPE::TYPE_CIRCLE || l->type == LENS_TYPE::TYPE_LINE){
				std::vector<std::vector<float3>> lensContour = ((CircleLens*)l)->Get3DContour(eyeWorld, false);

				glColor3f(0.39f, 0.89f, 0.26f);
				for (int i = 0; i < 1; i++){
					glBegin(GL_LINE_LOOP);
					for (auto v : lensContour[i]){
						glVertex3f(v.x, v.y, v.z);
					}
					glEnd();
				}

				glColor3f(0.82f, 0.31f, 0.67f);
				for (int i = 2; i < 3; i++){
					glBegin(GL_LINE_LOOP);
					for (auto v : lensContour[i]){
						glVertex3f(v.x, v.y, v.z);
					}
					glEnd();
				}

				//glBegin(GL_LINES);
				//glColor3f(0.2f, 1.0f, 0.2f);
				//for (auto v : lensContour[lensContour.size() - 2]){
				//	glVertex3f(v.x, v.y, v.z);
				//}
				//glColor3f(1.0f, 0.2f, 0.2f);
				//for (auto v : lensContour[lensContour.size() - 1]){
				//	glVertex3f(v.x, v.y, v.z);
				//}
				//glEnd();

			}
		}
		glPopAttrib();

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
		glPopAttrib();

	}


}
void LensRenderable::AddCircleLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new CircleLens(winSize.y * 0.1, actor->DataCenter());
	lenses.push_back(l);
	l->justChanged = true;
	actor->UpdateGL();
}

void LensRenderable::AddLineLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new LineLens(actor->DataCenter(), 0.3);
	lenses.push_back(l);
	l->justChanged = true;
	actor->UpdateGL();
	actor->SetInteractMode(INTERACT_MODE::MODIFYING_LENS);
}

void LensRenderable::AddLineLens3D()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new LineLens3D(actor->DataCenter(), 0.3);
	lenses.push_back(l);
	l->justChanged = true;
	actor->UpdateGL();
	actor->SetInteractMode(INTERACT_MODE::MODIFYING_LENS);
}

void LensRenderable::AddCurveBLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new CurveBLens(winSize.y * 0.1, actor->DataCenter());
	lenses.push_back(l);
	l->justChanged = true;
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

float LensRenderable::GetBackLensFocusRatio()
{
	float ret = 1;
	if (lenses.size() > 0)
		ret = lenses.back()->focusRatio;
	return ret;
}

float LensRenderable::GetBackLensObjectRadius()
{
	float ret = 0;
	if (lenses.size() > 0 && lenses.back()->type==LENS_TYPE::TYPE_CIRCLE)
		ret = ((CircleLens*)lenses.back())->objectRadius;
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
		if (l->PointInsideFullLens(x, y, modelview, projection, winSize.x, winSize.y)) {
			ret = true;
			break;
		}
	}
	return ret;
}

void LensRenderable::UpdateLensTwoFingers(int2 p1, int2 p2)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	if (lenses.size() > 0) {
		lenses.back()->ChangeLensTwoFingers(p1, p2, modelview, projection, winSize.x, winSize.y);
		lenses.back()->justChanged = true;
		actor->UpdateGL();
	}
}


bool LensRenderable::TwoPointsInsideALens(int2 p1, int2 p2)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	bool ret = false;
	for (int i = 0; i < lenses.size(); i++) {
		Lens* l = lenses[i];
		if (l->PointInsideLens(p1.x, p1.y, modelview, projection, winSize.x, winSize.y)
			&& l->PointInsideLens(p2.x, p2.y, modelview, projection, winSize.x, winSize.y)) {
			ret = true;
			break;
		}
	}
	return ret;
}


bool LensRenderable::OnLensInnerBoundary(int2 p1, int2 p2)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	bool ret = false;
	for (int i = 0; i < lenses.size(); i++) {
		Lens* l = lenses[i];
		if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE){
			if (l->PointOnInnerBoundary(p1.x, p1.y, modelview, projection, winSize.x, winSize.y)
				&& l->PointOnInnerBoundary(p2.x, p2.y, modelview, projection, winSize.x, winSize.y)) {
				ret = true;
				break;
			}
		}
		else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE){
			if (l->PointOnObjectInnerBoundary(p1.x, p1.y, modelview, projection, winSize.x, winSize.y)
				&& l->PointOnObjectInnerBoundary(p2.x, p2.y, modelview, projection, winSize.x, winSize.y)) {
				ret = true;
				break;
			}
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
			else if (l->type == LENS_TYPE::TYPE_LINE) {
				((LineLens *)l)->ctrlPoint1Abs = make_float2(x, y);
				((LineLens *)l)->ctrlPoint2Abs = make_float2(x, y);
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
			else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE && l->PointOnInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
				actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE);
				pickedLens = i;
				break;
			}
			else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE && l->PointOnOuterBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
				actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE);
				pickedLens = i;
				break;
			}
			else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE && l->PointOnObjectInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
				std::cout << "on bound" << std::endl;
				actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE);
				pickedLens = i;
				break;
			}
			else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE && l->PointOnObjectOuterBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
				std::cout << "on outer bound" << std::endl;
				actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE);
				pickedLens = i;
				break;
			}
			///!!! need to modify  for OBJECT_SPACE too!!!
			//else if (actor->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE && l->PointInsideLens(x, y, modelview, projection, winSize.x, winSize.y)) {
			//	actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_DEPTH);
			//	pickedLens = i;
			//	break;
			//}
			//else if (actor->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE && l->PointInsideObjectLens(x, y, modelview, projection, winSize.x, winSize.y)) {
			//	actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_DEPTH);
			//	pickedLens = i;
			//	break;
			//}
		}
		break;
	}

	}
	//return insideAnyLens;
	lastPt = make_int2(x, y);
	//std::cout << lastPt.x << ", " << lastPt.y << std::endl;
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
		else if (l->type == LENS_TYPE::TYPE_LINE) {
			((LineLens*)l)->FinishConstructing(modelview, projection, winSize.x, winSize.y);
			actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
			l->justChanged = true;
		}
	}
	else {
		if (actor->GetInteractMode() == INTERACT_MODE::MOVE_LENS || actor->GetInteractMode() == INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE || actor->GetInteractMode() == INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE || actor->GetInteractMode() == INTERACT_MODE::TRANSFORMATION){
			if (lenses.size() > 0){
				Lens* l = lenses[lenses.size() - 1];
				l->justChanged = true;
			}
		}
		if (actor->GetInteractMode() == INTERACT_MODE::MOVE_LENS && isSnapToFeature){
			/*// !!! DON'T DELETE !!!
			//these code will be process later
			GlyphRenderable* glyphRenderable = (GlyphRenderable*)actor->GetRenderable("glyph");
			Lens* l = lenses[lenses.size() - 1];
			float3 center = make_float3(l->GetCenter());
			int resid = -1;
			if (glyphRenderable->findClosetFeature(center, snapPos, resid))
			{
				l->SetCenter(snapPos);

				PolyRenderable* r1 = (PolyRenderable*)actor->GetRenderable("ventricles");
				PolyRenderable* r2 = (PolyRenderable*)actor->GetRenderable("tumor1");
				PolyRenderable* r3 = (PolyRenderable*)actor->GetRenderable("tumor2");
				if (resid == 1){
					r1->isSnapped = true;
					r2->isSnapped = false;
					r3->isSnapped = false;
				}
				else if (resid == 2){
					r1->isSnapped = false;
					r2->isSnapped = true;
					r3->isSnapped = false;
				}
				else if (resid == 3){
					r1->isSnapped = false;
					r2->isSnapped = false;
					r3->isSnapped = true;
				}
			}
			*/
		}
	}
	actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
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
		else if (l->type == LENS_TYPE::TYPE_LINE){
			((LineLens *)l)->ctrlPoint2Abs = make_float2(x, y);
			((LineLens*)l)->UpdateInfo(modelview, projection, winSize.x, winSize.y);
			l->justChanged = true;
		}
		break;
	}
	case INTERACT_MODE::MOVE_LENS:
	{
		//lenses[pickedLens]->x += (x - lastPt.x);
		//lenses[pickedLens]->y += (y - lastPt.y);
		float2 center = lenses[pickedLens]->GetCenterScreenPos(modelview, projection, winSize.x, winSize.y);
		lenses[pickedLens]->UpdateCenterByScreenPos(
			//center.x + (x - lastPt.x), center.y + (y - lastPt.y)
			x , y
			, modelview, projection, winSize.x, winSize.y);
		if (isSnapToGlyph){
			/*// !!! DON'T DELETE !!!
			//these code will be process later
			DeformGlyphRenderable* glyphRenderable = (DeformGlyphRenderable*)actor->GetRenderable("glyph");
			glyphRenderable->findClosetGlyph(make_float3(lenses[pickedLens]->GetCenter()));
			*/
		}
		else if (isSnapToFeature){
			/*// !!! DON'T DELETE !!!
			//these code will be process later			
			GlyphRenderable* glyphRenderable = (GlyphRenderable*)actor->GetRenderable("glyph");
			snapPos;
			int resid=-1;
			glyphRenderable->findClosetFeature(make_float3(lenses[pickedLens]->GetCenter()), snapPos, resid);

			PolyRenderable* r1 = (PolyRenderable*)actor->GetRenderable("ventricles");
			PolyRenderable* r2 = (PolyRenderable*)actor->GetRenderable("tumor1");
			PolyRenderable* r3 = (PolyRenderable*)actor->GetRenderable("tumor2");
			if (resid == 1){
				r1->isSnapped = true;
				r2->isSnapped = false;
				r3->isSnapped = false;
			}
			else if (resid == 2){
				r1->isSnapped = false;
				r2->isSnapped = true;
				r3->isSnapped = false;
			}
			else if (resid == 3){
				r1->isSnapped = false;
				r2->isSnapped = false;
				r3->isSnapped = true;
			}
			*/
		}
		lenses[pickedLens]->justChanged = true;
		break;
	}
	case INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE:
	{
		if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE)
			lenses[pickedLens]->ChangeLensSize(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);
		else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE)
			lenses[pickedLens]->ChangeObjectLensSize(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);
		break;
	}
	case INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE:
	{												   
		if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE)
			lenses[pickedLens]->ChangefocusRatio(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);
		else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE)
			lenses[pickedLens]->ChangeObjectFocusRatio(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);
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
			l->justChanged = true;
		}
	}
	return insideAnyLens;
}


void LensRenderable::SnapLastLens()
{
	if (lenses.size() > 0){
		Lens* l = lenses[lenses.size() - 1];
		l->SetCenter(snapPos);
	}
}

void LensRenderable::ChangeLensDepth(float v)
{
	//float scaleFactor = totalScaleFactor > 1 ? 1 : -1;
	if (lenses.size() == 0) return;
	lenses.back()->ChangeClipDepth(v, &matrix_mv.v[0].x, &matrix_pj.v[0].x);

	lenses.back()->justChanged = true;
	actor->UpdateGL();
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
	if (lenses.size()<1)
		return;
	if (INTERACT_MODE::MODIFY_LENS_DEPTH == actor->GetInteractMode()){
		//actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_DEPTH);
		//std::cout << "totalScaleFactor:" << totalScaleFactor << std::endl;
		float scaleFactor = totalScaleFactor > 1 ? 1 : -1;
		lenses[pickedLens]->ChangeClipDepth(scaleFactor, &matrix_mv.v[0].x, &matrix_pj.v[0].x);
		lenses[pickedLens]->justChanged = true;
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
		l->justChanged = true;
	}
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

void LensRenderable::SlotTwoHandChanged(float3 l, float3 r)
{
	//std::cout << "two hands..." << std::endl;
	if (lenses.size() > 0){
		ChangeLensCenterbyLeap((l+r) * 0.5);
		if (LENS_TYPE::TYPE_CIRCLE == lenses.back()->type){
			((CircleLens*)lenses.back())->radius = length(l-r) * 0.5;
			((CircleLens*)lenses.back())->justChanged = true;
		}
		actor->UpdateGL();
	}
}


void LensRenderable::ChangeLensCenterbyLeap(float3 p)
{
	//interaction box
	//https://developer.leapmotion.com/documentation/csharp/devguide/Leap_Coordinate_Mapping.html
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
		((DeformGLWidget*)actor)->GetDepthRange(depthRange);
		//pScreen.z = clamp((1.0f - aa * , 0.0f, 1.0f);
		//std::cout << "bb:" << clamp((1.0 - (p.z + 73.5f) / 147.0f), 0.0f, 1.0f) << std::endl;

		//std::cout << "leapPos:" << leapPos.x << "," << leapPos.y << "," << leapPos.z << std::endl;
		bool usingVR = false;
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
	}
}


void LensRenderable::SlotOneHandChanged(float3 p)
{
	//std::cout << "one hand..." << std::endl;
	if (lenses.size() > 0){
		ChangeLensCenterbyLeap(p);
		//interaction box
		//https://developer.leapmotion.com/documentation/csharp/devguide/Leap_Coordinate_Mapping.html
		(lenses.back())->justChanged = true;
		actor->UpdateGL();
	}
}
