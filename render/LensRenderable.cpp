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


void LensRenderable::draw(float modelview[16], float projection[16])
{
	RecordMatrix(modelview, projection);
	
	if (!visible)
		return;

	if (highlightingCenter){
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
			GLSphere sphere(1.0, 2);
			glColor4f(1.0f, 1.0f, 1.0f, 0.9f);
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glTranslatef(l->c.x, l->c.y, l->c.z);
			glScalef(0.1, 0.1, 0.1);
			glVertexPointer(3, GL_FLOAT, 0, sphere.GetVerts());
			glDrawArrays(GL_QUADS, 0, sphere.GetNumVerts());
			glPopMatrix();
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

			glDisableClientState(GL_VERTEX_ARRAY);

			glMatrixMode(GL_PROJECTION);
			glPopMatrix();
			glMatrixMode(GL_MODELVIEW);
			glPopMatrix();

		}
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

		for (int i = 0; i < lenses.size(); i++) {
			Lens* l = lenses[i];
			if (l->type == LENS_TYPE::TYPE_CIRCLE){ //same with screen space
				glMatrixMode(GL_PROJECTION);
				glPushMatrix();
				glLoadIdentity();
				glOrtho(0.0, winSize.x - 1, 0.0, winSize.y - 1, -1, 1);

				glMatrixMode(GL_MODELVIEW);
				glPushMatrix();
				glLoadIdentity();

				glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);
				glLineWidth(4);

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

				glPopAttrib();
				//restore the original 3D coordinate system
				glMatrixMode(GL_PROJECTION);
				glPopMatrix();
				glMatrixMode(GL_MODELVIEW);
				glPopMatrix();
			}
			else if (l->type == LENS_TYPE::TYPE_LINE){
				if (l->isConstructing){
					if (l->isConstructedFromLeap){ //when using leap motion
						glMatrixMode(GL_PROJECTION);
						glPushMatrix();
						glLoadIdentity();
						glLoadMatrixf(projection);
						glMatrixMode(GL_MODELVIEW);
						glPushMatrix();
						glLoadIdentity();
						glLoadMatrixf(modelview);

						glLineWidth(4);

						glColor3f(0.39f, 0.89f, 0.26f);
			
						std::vector<float3> PointsForIncisionBack = ((LineLens3D*)l)->GetCtrlPoints3DForRendering(modelview, projection, winSize.x, winSize.y);
						glBegin(GL_LINES);
						for (auto v : PointsForIncisionBack)
							glVertex3f(v.x, v.y, v.z);
						glEnd();

						glMatrixMode(GL_PROJECTION);
						glPopMatrix();
						glMatrixMode(GL_MODELVIEW);
						glPopMatrix();
					}
					else{ //when NOT using leap motion
						glMatrixMode(GL_PROJECTION);
						glPushMatrix();
						glLoadIdentity();
						glOrtho(0.0, winSize.x - 1, 0.0, winSize.y - 1, -1, 1);

						glMatrixMode(GL_MODELVIEW);
						glPushMatrix();
						glLoadIdentity();

						glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);
						glLineWidth(4);

						std::vector<float2> lensContour = l->GetCtrlPointsForRendering(modelview, projection, winSize.x, winSize.y);
						glColor3f(0.39f, 0.89f, 0.26f);
						glBegin(GL_LINES);
						for (auto v : lensContour)
							glVertex2f(v.x, v.y);
						glEnd();

						glColor3f(0.82f, 0.31f, 0.67f);
						std::vector<float2> lensOuterContour = l->GetOuterContour(modelview, projection, winSize.x, winSize.y);
						glBegin(GL_LINE_LOOP);
						for (auto v : lensOuterContour)
							glVertex2f(v.x, v.y);
						glEnd();

						glPopAttrib();
						//restore the original 3D coordinate system
						glMatrixMode(GL_PROJECTION);
						glPopMatrix();
						glMatrixMode(GL_MODELVIEW);
						glPopMatrix();
					}
				}
				else //finished Constructing
				{
					glMatrixMode(GL_PROJECTION);
					glPushMatrix();
					glLoadIdentity();
					glLoadMatrixf(projection);
					glMatrixMode(GL_MODELVIEW);
					glPushMatrix();
					glLoadIdentity();
					glLoadMatrixf(modelview);

			

					if(highlightingCuboidFrame)
						glLineWidth(8);
					else
						glLineWidth(4);

					glColor3f(0.82f, 0.31f, 0.67f);
					std::vector<float3> PointsForContourOuterBack = ((LineLens3D*)l)->GetOuterContourBackBase();
					std::vector<float3> PointsForContourOuterFront = ((LineLens3D*)l)->GetOuterContourFrontBase();
					glBegin(GL_LINE_LOOP);
					for (auto v : PointsForContourOuterBack)
						glVertex3f(v.x, v.y, v.z);
					glEnd();
					glBegin(GL_LINE_LOOP);
					for (auto v : PointsForContourOuterFront)
						glVertex3f(v.x, v.y, v.z);
					glEnd();
					glBegin(GL_LINES);
					for (int i = 0; i < PointsForContourOuterBack.size(); i++){
						float3 v = PointsForContourOuterBack[i], v2 = PointsForContourOuterFront[i];
						glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);
					}
					glEnd();

					if (highlightingMajorSide){
						glLineWidth(8);
						glColor3f(0.82f, 0.31f, 0.67f);
						glBegin(GL_LINES);
						float3 v = PointsForContourOuterBack[0], v2 = PointsForContourOuterBack[3]; glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);
						v = PointsForContourOuterBack[2], v2 = PointsForContourOuterBack[1]; glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);
						v = PointsForContourOuterFront[0], v2 = PointsForContourOuterFront[3]; glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);
						v = PointsForContourOuterFront[2], v2 = PointsForContourOuterFront[1]; glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);
						glEnd();

						glLineWidth(4);
						glColor3f(0.39f, 0.89f, 0.26f);
						glBegin(GL_LINES);
						v = ((LineLens3D*)l)->ctrlPoint3D1, v2 = ((LineLens3D*)l)->ctrlPoint3D2;
						glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);	
						glEnd();
					}
					else{
						glLineWidth(4);

						glColor3f(0.39f, 0.89f, 0.26f);
						if (l->isConstructedFromLeap){
							std::vector<float3> PointsForIncisionBack = ((LineLens3D*)l)->GetIncisionBack();
							glBegin(GL_LINES);
							for (auto v : PointsForIncisionBack)
								glVertex3f(v.x, v.y, v.z);
						}
						else{
							std::vector<float3> PointsForIncisionFront = ((LineLens3D*)l)->GetIncisionFront();
							glBegin(GL_LINES);
							for (auto v : PointsForIncisionFront)
								glVertex3f(v.x, v.y, v.z);
						}
						glEnd();

						if (highlightingMinorSide){
							glLineWidth(8);
							glColor3f(0.82f, 0.31f, 0.67f);
							glBegin(GL_LINES);
							float3 v = PointsForContourOuterBack[0], v2 = PointsForContourOuterBack[1]; glVertex3f(v.x, v.y, v.z);
							glVertex3f(v2.x, v2.y, v2.z);
							v = PointsForContourOuterBack[2], v2 = PointsForContourOuterBack[3]; glVertex3f(v.x, v.y, v.z);
							glVertex3f(v2.x, v2.y, v2.z);
							v = PointsForContourOuterFront[0], v2 = PointsForContourOuterFront[1]; glVertex3f(v.x, v.y, v.z);
							glVertex3f(v2.x, v2.y, v2.z);
							v = PointsForContourOuterFront[2], v2 = PointsForContourOuterFront[3]; glVertex3f(v.x, v.y, v.z);
							glVertex3f(v2.x, v2.y, v2.z);
							glEnd();
						}
					}

					glMatrixMode(GL_PROJECTION);
					glPopMatrix();
					glMatrixMode(GL_MODELVIEW);
					glPopMatrix();
				}
			}
		}
	}
	
}

void LensRenderable::AddCircleLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l;
	if (lastLensCenterRecorded){
		l = new CircleLens(winSize.y * 0.1, lastLensCenter);
	}
	else{
		l = new CircleLens(winSize.y * 0.1, actor->DataCenter());
	}
	lenses.push_back(l);
	l->justChanged = true;
	actor->UpdateGL();
}

void LensRenderable::AddLineLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l;
	if (lastLensCenterRecorded){
		l = new LineLens(lastLensCenter, 0.3);
	}
	else{
		l = new LineLens(actor->DataCenter(), 0.3);
	}
	lenses.push_back(l);
	//l->justChanged = true; //constructing first, then set justChanged
	actor->UpdateGL();
	actor->SetInteractMode(INTERACT_MODE::ADDING_LENS);
}

void LensRenderable::AddLineLens3D()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l;
	if (lastLensCenterRecorded){
		l = new LineLens3D(lastLensCenter, 0.3);
	}
	else{
		l = new LineLens3D(actor->DataCenter(), 0.3);
	}
	lenses.push_back(l);
	//l->justChanged = true; //constructing first, then set justChanged
	actor->UpdateGL();
	actor->SetInteractMode(INTERACT_MODE::ADDING_LENS);
}

void LensRenderable::AddCurveBLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new CurveBLens(winSize.y * 0.1, actor->DataCenter());
	lenses.push_back(l);
	//l->justChanged = true; //constructing first, then set justChanged
	actor->UpdateGL();
	actor->SetInteractMode(INTERACT_MODE::ADDING_LENS);
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


void LensRenderable::mousePress(int x, int y, int modifier)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	switch (actor->GetInteractMode())
	{
	case INTERACT_MODE::ADDING_LENS:
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
				actor->SetInteractMode(INTERACT_MODE::MOVE_LENS);
				pickedLens = i;
				break;
			}
			else if (l->PointOnOuterBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
				actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE);
				pickedLens = i;
				break;
			}
			else if (l->PointOnInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
				actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE);
				pickedLens = i;
				break;
			}
			/*
			else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE && l->PointOnOuterBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
				actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE);
				pickedLens = i;
				break;
			}
			else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE && l->PointOnInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
				actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE);
				pickedLens = i;
				break;
			}
			
			else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE){
				if (l->type == LENS_TYPE::TYPE_CIRCLE){
					if (l->PointOnObjectInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
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
				}
				else if (l->type == LENS_TYPE::TYPE_LINE){
					if (((LineLens3D*)l)->PointOnObjectOuterBoundaryMajorSide(x, y, modelview, projection, winSize.x, winSize.y)) {
						std::cout << "on major bound" << std::endl;
						actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE);
						pickedLens = i;
						break;
					}
					else if (((LineLens3D*)l)->PointOnObjectOuterBoundaryMinorSide(x, y, modelview, projection, winSize.x, winSize.y)) {
						std::cout << "on minor bound" << std::endl;
						actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE);
						pickedLens = i;
						break;
					}
				}
			}*/
		}
		break;
	}

	}
	//return insideAnyLens;
	lastPt = make_int2(x, y);
	//std::cout << lastPt.x << ", " << lastPt.y << std::endl;
	highlightingCenter = true;
}

void LensRenderable::mouseRelease(int x, int y, int modifier)
{
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	if (INTERACT_MODE::ADDING_LENS == actor->GetInteractMode()) {
		Lens* l = lenses[lenses.size() - 1];
		if (l->type == LENS_TYPE::TYPE_LINE) {
			if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE)
			{
				((LineLens*)l)->FinishConstructing(modelview, projection, winSize.x, winSize.y);
				l->justChanged = true;
				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
			}
			else
			{
				((LineLens3D*)l)->FinishConstructing(modelview, projection, winSize.x, winSize.y, make_float3(0, 0, 0), make_float3(0, 0, 0));
				l->justChanged = true;
				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
			}
		}
		else if (l->type == LENS_TYPE::TYPE_CURVEB) {
			((CurveBLens *)l)->FinishConstructing(modelview, projection, winSize.x, winSize.y);
			l->justChanged = true;
			actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
		}
	}
	else {
		if (actor->GetInteractMode() == INTERACT_MODE::MOVE_LENS){
			if (lenses.size() > 0){
				Lens* l = lenses[lenses.size() - 1];
				l->justChanged = true;
			}
		}
		else if( actor->GetInteractMode() == INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE || actor->GetInteractMode() == INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE){
			if (lenses.size() > 0){
				Lens* l = lenses[lenses.size() - 1];
				l->justChanged = true;
			}
		}
		else if (actor->GetInteractMode() == INTERACT_MODE::TRANSFORMATION){
			//if (lenses.size() > 0){
			//	Lens* l = lenses[lenses.size() - 1];
			//	l->justChanged = true;
			//}
			;
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
	highlightingCenter = false;
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
	case INTERACT_MODE::ADDING_LENS:
	{
		Lens* l = lenses[lenses.size() - 1];
		if (l->type == LENS_TYPE::TYPE_CURVEB){
			((CurveBLens *)l)->AddCtrlPoint(x, y);
		}
		else if (l->type == LENS_TYPE::TYPE_LINE){
			((LineLens *)l)->ctrlPoint2Abs = make_float2(x, y);
			if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE)
				((LineLens*)l)->UpdateInfoFromCtrlPoints(modelview, projection, winSize.x, winSize.y);
		}
		break;
	}
	case INTERACT_MODE::MOVE_LENS:
	{
		//lenses[pickedLens]->x += (x - lastPt.x);
		//lenses[pickedLens]->y += (y - lastPt.y);
		float2 center = lenses[pickedLens]->GetCenterScreenPos(modelview, projection, winSize.x, winSize.y);
		float3 moveVec = lenses[pickedLens]->UpdateCenterByScreenPos(x , y
			, modelview, projection, winSize.x, winSize.y);

		lenses[pickedLens]->setJustMoved(moveVec);

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
		//lenses[pickedLens]->justChanged = true;
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
			l->ChangeClipDepth(delta, &matrix_mv.v[0].x, &matrix_pj.v[0].x);

		}
	}
	return insideAnyLens;
}


////////////////////////////  for leap ///////////////////////////////



inline float3 GetNormalizedLeapPos(float3 p)
{
	float3 leapPos;
	leapPos.x = clamp((p.x + 117.5) / 235.0, 0.0f, 1.0f);
	leapPos.y = clamp((p.y - 82.5) / 235.0, 0.0f, 1.0f);
	leapPos.z = clamp((p.z + 73.5f) / 147.0f, 0.0f, 1.0f);
	return leapPos;
}

float3 LensRenderable::GetTransferredLeapPos(float3 p)
{
	//float3 posMin, posMax;
	//actor->GetPosRange(posMin, posMax);
	//float3 leapPosCameraNormalized = GetNormalizedLeapPos(p);
	//float3 leapPosCamera = leapPosCameraNormalized*(posMax - posMin) + posMin;
	//GLfloat modelview[16];
	////GLfloat projection[16];
	//actor->GetModelview(modelview);
	////actor->GetProjection(projection);
	//float _invmv[16];
	//invertMatrix(modelview, _invmv);
	//float4 leapPos_f4 = Camera2Object(make_float4(leapPosCamera, 1), _invmv);
	//return make_float3(leapPos_f4);

	//float3 leapPosClipNormalized = GetNormalizedLeapPos(p);
	//float3 leapPosClip = leapPosClipNormalized*2 -make_float3(1,1,1);
	//leapPosClip.z = -leapPosClip.z;

	//GLfloat modelview[16];
	//GLfloat projection[16];
	//actor->GetModelview(modelview);
	//actor->GetProjection(projection);
	//float _invmv[16];
	//float _invpj[16];
	//invertMatrix(projection, _invpj);
	//invertMatrix(modelview, _invmv);

	//float4 leapPos_f4 = Clip2ObjectGlobal(make_float4(leapPosClip, 1), _invmv, _invpj);

	//return make_float3(leapPos_f4);

	//float3 posMin, posMax;
	//actor->GetPosRange(posMin, posMax);
	//float3 dataCenter = actor->DataCenter();
	//float3 leapPosNormalized = GetNormalizedLeapPos(p);
	//float3 leapPos = leapPosNormalized * 2 - make_float3(1, 1, 1);
	//GLfloat modelview[16];
	////GLfloat projection[16];
	//actor->GetModelview(modelview);
	////actor->GetProjection(projection);
	//float _invmv[16];
	////float _invpj[16];
	////invertMatrix(projection, _invpj);
	//invertMatrix(modelview, _invmv);
	//float4 camerax4 = Camera2Object(make_float4(1, 0, 0, 0), _invmv);
	//float4 cameray4 = Camera2Object(make_float4(0, 1, 0, 0), _invmv);
	//float4 cameraz4 = Camera2Object(make_float4(0, 0, 1, 0), _invmv);
	//float3 camerax = normalize(make_float3(camerax4));
	//float3 cameray = normalize(make_float3(cameray4));
	//float3 cameraz = normalize(make_float3(cameraz4));

	//return dataCenter + leapPos.x*camerax + leapPos.y*cameray - leapPos.z*cameraz;

	////Xin's method when not using VR
	float3 leapPosNormalized = GetNormalizedLeapPos(p);

	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);

	float2 depthRange;
	((DeformGLWidget*)actor)->GetDepthRange(depthRange);

	float leapClipx = leapPosNormalized.x * 2 - 1;
	float leapClipy = leapPosNormalized.y * 2 - 1; 
	float leapClipz = depthRange.x + (depthRange.y - depthRange.x) * (1.0 - leapPosNormalized.z);


	float _invmv[16];
	float _invpj[16];
	invertMatrix(projection, _invpj);
	invertMatrix(modelview, _invmv);

	return make_float3(Clip2ObjectGlobal(make_float4(leapClipx, leapClipy, leapClipz, 1.0), _invmv, _invpj));

}

void LensRenderable::SlotTwoHandChanged(float3 l, float3 r)
{
	//std::cout << "two hands..." << std::endl;
	if (lenses.size() > 0){
		ChangeLensCenterbyLeap(lenses.back(), (l + r) * 0.5);
		if (LENS_TYPE::TYPE_CIRCLE == lenses.back()->type){
			((CircleLens*)lenses.back())->radius = length(l-r) * 0.5;
			((CircleLens*)lenses.back())->justChanged = true;
		}
		actor->UpdateGL();
	}
}

void LensRenderable::ChangeLensCenterbyTransferredLeap(Lens *l, float3 p)
{
	if (DEFORM_MODEL::OBJECT_SPACE == ((DeformGLWidget*)actor)->GetDeformModel()){

		float3 newCenter = p;

		//std::cout << "indexLeap " << p.x << " " << p.y << " " << p.z << std::endl;
		//std::cout << "newCenter " << newCenter.x << " " << newCenter.y << " " << newCenter.z << std::endl;

		if (l->type == LENS_TYPE::TYPE_LINE){
			float3 moveDir = newCenter - l->c;
			l->SetCenter(newCenter);
			((LineLens3D*)l)->ctrlPoint3D1 += moveDir;
			((LineLens3D*)l)->ctrlPoint3D2 += moveDir;
		}
		else if (l->type == LENS_TYPE::TYPE_CIRCLE){

		}
	}
	else{
		
	}

}


void LensRenderable::ChangeLensCenterbyLeap(Lens *l, float3 p)
{
	if (DEFORM_MODEL::OBJECT_SPACE == ((DeformGLWidget*)actor)->GetDeformModel()){

		float3 newCenter = GetTransferredLeapPos(p);

		std::cout << "indexLeap " << p.x << " " << p.y << " " << p.z << std::endl;
		std::cout << "newCenter " << newCenter.x << " " << newCenter.y << " " << newCenter.z << std::endl;

		if (l->type == LENS_TYPE::TYPE_LINE){
			float3 moveDir = newCenter - l->c;
			l->SetCenter(newCenter);
			((LineLens3D*)l)->ctrlPoint3D1 += moveDir;
			((LineLens3D*)l)->ctrlPoint3D2 += moveDir;
		}
		else if (l->type == LENS_TYPE::TYPE_CIRCLE){

		}
	}
	else{
		int2 winSize = actor->GetWindowSize();
		GLfloat modelview[16];
		GLfloat projection[16];
		actor->GetModelview(modelview);
		actor->GetProjection(projection);
		float3 pScreen;
		float3 leapPos = GetNormalizedLeapPos(p);
		const float aa = 0.02f;
		float2 depthRange;
		((DeformGLWidget*)actor)->GetDepthRange(depthRange);

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
		l->SetClipDepth(pScreen.z, &matrix_mv.v[0].x, &matrix_pj.v[0].x);
		l->UpdateCenterByScreenPos(pScreen.x, pScreen.y, modelview, projection, winSize.x, winSize.y);
	}

}


void LensRenderable::SlotOneHandChanged(float3 p)
{
	//std::cout << "one hand..." << std::endl;
	if (lenses.size() > 0){
		ChangeLensCenterbyLeap(lenses.back(), p);
		//interaction box
		//https://developer.leapmotion.com/documentation/csharp/devguide/Leap_Coordinate_Mapping.html
		(lenses.back())->justChanged = true;
		actor->UpdateGL();
	}
}

bool LensRenderable::SlotOneHandChanged_lc(float3 thumpLeap, float3 indexLeap, float4 &markerPos)
{
	//currently only work for line lens 3D
	//std::cout << "thumpLeap " << thumpLeap.x << " " << thumpLeap.y << " " << thumpLeap.z << std::endl;
	//std::cout << "indexLeap " << indexLeap.x << " " << indexLeap.y << " " << indexLeap.z << std::endl;


	float enterPichThr = 25, leavePinchThr = 35; //different threshold to avoid shaking
	float d = length(thumpLeap - indexLeap);
	
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);


	if (lenses.size() == 0){
		
		if (actor->GetInteractMode() == INTERACT_MODE::TRANSFORMATION && d < enterPichThr){

			Lens* l = new LineLens3D(actor->DataCenter(), 0.3);
			l->isConstructedFromLeap = true;

			lenses.push_back(l);
			//l->justChanged = true; //constructing first, then set justChanged
			actor->UpdateGL();
			actor->SetInteractMode(INTERACT_MODE::ADDING_LENS);
			float3 curPos = GetTransferredLeapPos(indexLeap);
			((LineLens3D *)l)->ctrlPoint3D1 = curPos;
			((LineLens3D *)l)->ctrlPoint3D2 = curPos;
			markerPos = make_float4(curPos, 1.0);

			return true;
		}
		else{
			//add global rotation

			float3 curPos = GetTransferredLeapPos(indexLeap);
			markerPos = make_float4(curPos, 1);
			return true;
		}
	}
	else{
		LineLens3D* l = (LineLens3D*)lenses.back();
		if (actor->GetInteractMode() == INTERACT_MODE::ADDING_LENS){
			if (d < leavePinchThr){
				l->ctrlPoint3D2 = GetTransferredLeapPos(indexLeap);
				float3 curPos = GetTransferredLeapPos(indexLeap);
				markerPos = make_float4(curPos, 1.0);
				return true;
			}
			else {
				float3 posMin, posMax;
				actor->GetPosRange(posMin, posMax);
				l->FinishConstructing3D(modelview, projection, winSize.x, winSize.y, posMin, posMax);

				l->justChanged = true;

				float3 curPos = GetTransferredLeapPos(indexLeap);
				markerPos = make_float4(curPos, 1.0);

				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				return true;
			}
		}
		else if (actor->GetInteractMode() == INTERACT_MODE::TRANSFORMATION)
		{
			if (d < enterPichThr){
				float3 curPos = GetTransferredLeapPos(indexLeap);
				markerPos = make_float4(curPos, 1.0);

				if (l->PointOnLensCenter3D(curPos, modelview, projection, winSize.x, winSize.y)){
					actor->SetInteractMode(INTERACT_MODE::MOVE_LENS);
					highlightingCenter = true;
					highlightingCuboidFrame = false;
					prevPos = curPos;
					prevPointOfLens = l->c;
				}
				else if (l->PointOnOuterBoundaryWallMajorSide3D(curPos, modelview, projection, winSize.x, winSize.y)){
					highlightingMajorSide = true;
					highlightingCuboidFrame = false;
					actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE);
					prevPos = curPos;
					if (length(curPos - l->ctrlPoint3D1) < length(curPos - l->ctrlPoint3D2)){
						prevPointOfLens = l->ctrlPoint3D1;
					}
					else{
						prevPointOfLens = l->ctrlPoint3D2;
					}
				}
				else if (l->PointOnOuterBoundaryWallMinorSide3D(curPos, modelview, projection, winSize.x, winSize.y)){
					highlightingMinorSide = true;
					highlightingCuboidFrame = false;
					actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE);
					prevPos = curPos;
					float3 pp1 = l->c - l->minorAxisGlobal*l->lSemiMinorAxisGlobal / l->focusRatio;
					float3 pp2 = l->c + l->minorAxisGlobal*l->lSemiMinorAxisGlobal / l->focusRatio;
					if (length(curPos - pp1) < length(curPos - pp2)){
						prevPointOfLens = pp1;
					}
					else{
						prevPointOfLens = pp2;
					}
				}
				return true;
			}
			else{
		
				float3 curPos = GetTransferredLeapPos(indexLeap);
				markerPos = make_float4(curPos, 1.0);
				float3 pmin, pmax;
				actor->GetPosRange(pmin, pmax);
				float3 difmin = curPos - pmin, difmax = curPos - pmax;
				if (min(difmin.x, min(difmin.y, difmin.z))<0 || max(difmax.x, max(difmax.y, difmax.z))>0){
					return false;
				}
				else{
					if (l->PointInCuboidRegion3D(curPos, modelview, projection, winSize.x, winSize.y))
						highlightingCuboidFrame = true;
					else
						highlightingCuboidFrame = false;
				}
				return true;
			}
		}
		else if (actor->GetInteractMode() == INTERACT_MODE::MOVE_LENS){
			float3 curPos = GetTransferredLeapPos(indexLeap);
			markerPos = make_float4(curPos, 1.0); 
			
			if (d > leavePinchThr){
				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				(lenses.back())->justChanged = true;
				highlightingCenter = false;
			}
			else{
				//ChangeLensCenterbyLeap(lenses.back(), indexLeap);
				float3 moveDir = curPos - prevPos;
				ChangeLensCenterbyTransferredLeap(lenses.back(), prevPointOfLens + moveDir);
			}
			return true;
		}
		else if (actor->GetInteractMode() == INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE){
			float3 curPos = GetTransferredLeapPos(indexLeap);
			markerPos = make_float4(curPos, 1.0);

			if (d > leavePinchThr){
				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				(lenses.back())->justChanged = true;
				highlightingMajorSide = false;
			}
			else{
				float3 moveDir = curPos - prevPos;
				if (length(curPos - l->ctrlPoint3D1) < length(curPos - l->ctrlPoint3D2)){
					l->ctrlPoint3D1 = prevPointOfLens + moveDir;
				}
				else{
					l->ctrlPoint3D2 = prevPointOfLens + moveDir;
				}
			}
			return true;
		}
		else if (actor->GetInteractMode() == INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE){
			float3 curPos = GetTransferredLeapPos(indexLeap);
			markerPos = make_float4(curPos, 1.0);


			float3 moveDir = curPos - prevPos;
			float3 pp = prevPointOfLens + moveDir;
			float newMinorDis = abs(dot(pp - l->c, l->minorAxisGlobal));

			if (d > leavePinchThr || newMinorDis <= l->lSemiMinorAxisGlobal){
				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				(lenses.back())->justChanged = true;
				highlightingMinorSide = false;
			}
			else{
				//float3 moveDir = curPos - prevPos;
				//float3 pp = prevPointOfLens + moveDir;
				l->focusRatio = l->lSemiMinorAxisGlobal / newMinorDis;
					
				//float3 moveDir = curPos - prevPos;
				//float3 pp1 = l->c - l->minorAxisGlobal*l->lSemiMinorAxisGlobal / l->focusRatio;
				//float3 pp2 = l->c + l->minorAxisGlobal*l->lSemiMinorAxisGlobal / l->focusRatio;
				//if (length(curPos - pp1) < length(curPos - pp2)){
				//	pp1 = prevPointOfLens + moveDir;
				//	l->focusRatio = abs(dot(pp1 - l->c, l->minorAxisGlobal)) / l->lSemiMinorAxisGlobal;
				//}
				//else{
				//	pp2 = prevPointOfLens + moveDir;
				//	l->focusRatio = abs(dot(pp2 - l->c, l->minorAxisGlobal)) / l->lSemiMinorAxisGlobal;
				//}
			}
			return true;
		}
		else{
			//do nothing
			return false;
		}
	}
}



//////////////////////////////////	//for keyboard /////////////////////

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

void LensRenderable::SlotFocusSizeChanged(int v)
{
	if (lenses.size() > 0){
		lenses.back()->SetFocusRatio((49 - v) * 0.02 * 0.8 + 0.2);
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
		lastLensCenter = make_float3(lenses.back()->GetCenter());
		lastLensCenterRecorded = true;
		lenses.pop_back();
	}
	actor->UpdateGL();
}



//////////////////////////////////	//for touch screen /////////////////////

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

void LensRenderable::ChangeLensDepth(float v)
{
	//float scaleFactor = totalScaleFactor > 1 ? 1 : -1;
	if (lenses.size() == 0) return;
	lenses.back()->ChangeClipDepth(v, &matrix_mv.v[0].x, &matrix_pj.v[0].x);

	//lenses.back()->justChanged = true;
	actor->UpdateGL();
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


void LensRenderable::SaveState(const char* filename)
{
	std::ofstream myfile;
	myfile.open(filename);

	if (lenses.size() == 0)
		return;
	LineLens3D* l = (LineLens3D*)lenses.back();
	myfile << l->c.x << " " << l->c.y << " " << l->c.z << std::endl;
	
	myfile << l->ctrlPoint1Abs.x << " " << l->ctrlPoint1Abs.y << std::endl;
	myfile << l->ctrlPoint2Abs.x << " " << l->ctrlPoint2Abs.y << std::endl;
	myfile << l->focusRatio;
	myfile.close();
}

void LensRenderable::LoadState(const char* filename)
{
	std::ifstream ifs(filename, std::ifstream::in);
	if (ifs.is_open()) {
		if (lenses.size() == 0)
			return;
		LineLens3D* l = (LineLens3D*)lenses.back();

		float3 _c;
		ifs >> _c.x >> _c.y >> _c.z;
		l->SetCenter(_c);

		ifs >> l->ctrlPoint1Abs.x;
		ifs >> l->ctrlPoint1Abs.y;
		ifs >> l->ctrlPoint2Abs.x;
		ifs >> l->ctrlPoint2Abs.y;

		ifs >> l->focusRatio;

		ifs.close();
		l->justChanged = true;
	}
}
