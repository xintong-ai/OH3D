#include "LensRenderable.h"
#include "Lens.h"
#include "DeformGLWidget.h"
#include "GLSphere.h"
//#include "PolyRenderable.h"

#include <fstream>

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

LensRenderable::LensRenderable(std::vector<Lens*>* _l)
{
	lenses = _l;
	lensCenterSphere = new SolidSphere(0.1, 12, 24);
}

LensRenderable::~LensRenderable()
{
	delete lensCenterSphere;
}


void LensRenderable::init()
{
}


void LensRenderable::draw(float modelview[16], float projection[16])
{
	RecordMatrix(modelview, projection);
	
	if (!visible)
		return;

#ifdef USE_NEW_LEAP
	//draw cursor currently
	//add transparency later
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(projection);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(modelview);


	glLineWidth(32);
	glColor3f(0.82f, 0.0f, 0.0f);

	QMatrix4x4 q_modelview = QMatrix4x4(modelview);
	q_modelview = q_modelview.transposed();

	for (int i = 0; i < activedCursors; i++){
		QVector3D q_IndicatorDir;
		if (i == 0)
			q_IndicatorDir = QVector3D(QMatrix4x4((q_modelview.inverted()).normalMatrix()).map(QVector4D(-1, 1, 0, 0))).normalized();
		else
			q_IndicatorDir = QVector3D(QMatrix4x4((q_modelview.inverted()).normalMatrix()).map(QVector4D(1, 1, 0, 0))).normalized();
		float3 curVec = make_float3(q_IndicatorDir.x(), q_IndicatorDir.y(), q_IndicatorDir.z());

		if (cursorColor[i] == 0){
			glColor3f(1.0f, 0.0f, 0.0f);
		}
		else if (cursorColor[i] == 1){
			glColor3f(1.0f, 1.0f, 1.0f);
		}
		else if (cursorColor[i] == 2){
			glColor3f(0.0f, 0.0f, 1.0f);
		}

		glBegin(GL_LINES);
		glVertex3f(cursorPos[i].x, cursorPos[i].y, cursorPos[i].z);
		glVertex3f(cursorPos[i].x - curVec.x, cursorPos[i].y - curVec.y, cursorPos[i].z - curVec.z);
		glEnd();

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_POINT_SMOOTH);
		glPointSize(28.0);
		glBegin(GL_POINTS);
		glVertex3f(cursorPos[i].x, cursorPos[i].y, cursorPos[i].z);
		glEnd();
	}


	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
#endif


	if (highlightingCenter){
		for (int i = 0; i < lenses->size(); i++) {
			Lens* l = (*lenses)[i];
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
	
	if (DEFORM_MODEL::OBJECT_SPACE == ((DeformGLWidget*)actor)->GetDeformModel()){
		int2 winSize = actor->GetWindowSize();

		for (int i = 0; i < lenses->size(); i++) {
			Lens* l = (*lenses)[i];
			if (l->type == LENS_TYPE::TYPE_LINE){
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
					glDisable(GL_BLEND);
					glMatrixMode(GL_PROJECTION);
					glPushMatrix();
					glLoadIdentity();
					glLoadMatrixf(projection);
					glMatrixMode(GL_MODELVIEW);
					glPushMatrix();
					glLoadIdentity();
					glLoadMatrixf(modelview);

					if (highlightingCuboidFrame)
						glLineWidth(8);
					else
						glLineWidth(4);
					std::vector<float3> PointsForContourOuterCenter = ((LineLens3D*)l)->GetOuterContourCenterFace();
					std::vector<float3> PointsForContourOuterFront = ((LineLens3D*)l)->GetOuterContourFrontFace();


					glColor3f(0.82f, 0.31f, 0.67f);
					PointsForContourOuterCenter = ((LineLens3D*)l)->GetOuterContourCenterFace();
					PointsForContourOuterFront = ((LineLens3D*)l)->GetOuterContourFrontFace();
						glBegin(GL_LINE_LOOP);
						for (auto v : PointsForContourOuterCenter)
							glVertex3f(v.x, v.y, v.z);
						glEnd();
					
					glBegin(GL_LINE_LOOP);
					for (auto v : PointsForContourOuterFront)
						glVertex3f(v.x, v.y, v.z);
					glEnd();	
						glBegin(GL_LINES);
						for (int i = 0; i < PointsForContourOuterCenter.size(); i++){
							float3 v = PointsForContourOuterCenter[i], v2 = PointsForContourOuterFront[i];
							glVertex3f(v.x, v.y, v.z);
							glVertex3f(v2.x, v2.y, v2.z);
						}
						glEnd();
					if (drawFullRetractor){
						std::vector<float3> PointsForContourOuterBack = ((LineLens3D*)l)->GetOuterContourBackFace();
						glBegin(GL_LINE_LOOP);
						for (auto v : PointsForContourOuterBack)
							glVertex3f(v.x, v.y, v.z);
						glEnd();

						glBegin(GL_LINES);
						for (int i = 0; i < PointsForContourOuterCenter.size(); i++){
							float3 v = PointsForContourOuterCenter[i], v2 = PointsForContourOuterBack[i];
							glVertex3f(v.x, v.y, v.z);
							glVertex3f(v2.x, v2.y, v2.z);
						}
						glEnd();
					}

					if (highlightingMajorSide){
						glLineWidth(8);
						glColor3f(0.82f*1.1, 0.31f*1.1, 0.67f*1.1);
						glBegin(GL_LINES);
						float3 v = PointsForContourOuterCenter[0], v2 = PointsForContourOuterCenter[3]; glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);
						v = PointsForContourOuterCenter[2], v2 = PointsForContourOuterCenter[1]; glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);
						v = PointsForContourOuterFront[0], v2 = PointsForContourOuterFront[3]; glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);
						v = PointsForContourOuterFront[2], v2 = PointsForContourOuterFront[1]; glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);
						glEnd();
					}
					else if (highlightingMinorSide){
						glLineWidth(8);
						glColor3f(0.82f*1.1, 0.31f*1.1, 0.67f*1.1);
						glBegin(GL_LINES);
						float3 v = PointsForContourOuterCenter[0], v2 = PointsForContourOuterCenter[1]; glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);
						v = PointsForContourOuterCenter[2], v2 = PointsForContourOuterCenter[3]; glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);
						v = PointsForContourOuterFront[0], v2 = PointsForContourOuterFront[1]; glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);
						v = PointsForContourOuterFront[2], v2 = PointsForContourOuterFront[3]; glVertex3f(v.x, v.y, v.z);
						glVertex3f(v2.x, v2.y, v2.z);
						glEnd();		
					}

					//incision
					glLineWidth(4);
					glColor3f(0.39f, 0.89f, 0.26f);
					std::vector<float3> PointsForIncision;
					if (l->isConstructedFromLeap && actor->GetInteractMode() == INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE){
						PointsForIncision.push_back(((LineLens3D*)l)->ctrlPoint3D1);
						PointsForIncision.push_back(((LineLens3D*)l)->ctrlPoint3D2);
					}
					else if (l->isConstructedFromLeap || drawInsicionOnCenterFace){
						PointsForIncision = ((LineLens3D*)l)->GetIncisionCenter();
					}
					else{
						PointsForIncision = ((LineLens3D*)l)->GetIncisionFront();
					}
					glBegin(GL_LINES);
					for (auto v : PointsForIncision)
						glVertex3f(v.x, v.y, v.z);
					glEnd();
					
					glMatrixMode(GL_PROJECTION);
					glPopMatrix();
					glMatrixMode(GL_MODELVIEW);
					glPopMatrix();
				}
			}
			else if (l->type == LENS_TYPE::TYPE_CIRCLE){
				glDisable(GL_BLEND);
				glMatrixMode(GL_PROJECTION);
				glPushMatrix();
				glLoadIdentity();
				glLoadMatrixf(projection);
				glMatrixMode(GL_MODELVIEW);
				glPushMatrix();
				glLoadIdentity();
				glLoadMatrixf(modelview); 

				std::vector<std::vector<float3>> lensContour = ((CircleLens3D*)l)->Get3DContour(modelview);

				glLineWidth(4);
				glColor3f(0.39f, 0.89f, 0.26f);
				for (int i = 0; i < 1; i++){
					glBegin(GL_LINE_LOOP);
					for (auto v : lensContour[i]){
						glVertex3f(v.x, v.y, v.z);
					}
					glEnd();
				}
				glColor3f(0.82f, 0.31f, 0.67f);
				for (int i = 1; i < 2; i++){
					glBegin(GL_LINE_LOOP);
					for (auto v : lensContour[i]){
						glVertex3f(v.x, v.y, v.z);
					}
					glEnd();
				}
				glMatrixMode(GL_PROJECTION);
				glPopMatrix();
				glMatrixMode(GL_MODELVIEW);
				glPopMatrix();
			}
		}
	}
	else if (DEFORM_MODEL::SCREEN_SPACE == ((DeformGLWidget*)actor)->GetDeformModel()){
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
		for (int i = 0; i < lenses->size(); i++) {
			Lens* l = (*lenses)[i];

			std::vector<float2> lensContour = l->GetInnerContour(modelview, projection, winSize.x, winSize.y);
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
			if (l->type == LENS_TYPE::TYPE_CURVE) {
				glLineWidth(2);
				glColor3f(0.9f, 0.9f, 0.2f);
				std::vector<float2> lensExtraRendering2 = ((CurveLens *)l)->GetCenterLineForRendering(modelview, projection, winSize.x, winSize.y);
				//glBegin(GL_POINTS);
				glBegin(GL_LINE_STRIP);
				for (auto v : lensExtraRendering2)
					glVertex2f(v.x, v.y);
				glEnd();

				glColor3f(0.39f, 0.89f, 0.26f);
				std::vector<float2> lensContour = l->GetInnerContour(modelview, projection, winSize.x, winSize.y);
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


				std::vector<float2> pp = ((CurveLens *)l)->posOffsetCtrlPoints;
				std::vector<float2> nn = ((CurveLens *)l)->negOffsetCtrlPoints;
				//
				//vector<float2> pb = ((CurveLens *)l)->posOffsetBezierPoints;
				//vector<float2> nb = ((CurveLens *)l)->negOffsetBezierPoints;
				std::vector<float2> subp = ((CurveLens *)l)->subCtrlPointsPos;
				//vector<float2> subn = ((CurveLens *)l)->subCtrlPointsNeg;

				float2 center = ((CurveLens *)l)->GetCenterScreenPos(modelview, projection, winSize.x, winSize.y);
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
					for (int i = 0; i < pp.size(); i += 2){
						float2 v = pp[i];
						glVertex2f(v.x, v.y);
					}
					for (int i = 0; i < nn.size(); i++){
						float2 v = nn[i];
						glVertex2f(v.x, v.y);
					}


					glColor3f(0.0, 0.0f, 1.0);
					for (int i = 0; i < subp.size(); i += 2){
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
				//std::vector<float2> lensExtraRendering = ((CurveLens *)l)->GetCtrlPointsForRendering(modelview, projection, winSize.x, winSize.y);	
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
}

void LensRenderable::AddCircleLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l;
	if (lastLensCenterRecorded){
		l = new CircleLens(winSize.y * 0.2, lastLensCenter);
	}
	else{
		l = new CircleLens(winSize.y * 0.2, actor->DataCenter());
	}
	lenses->push_back(l);
	l->justChanged = true;
	actor->UpdateGL();
}

void LensRenderable::AddCircleLens3D()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l;
	
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);

	if (lastLensCenterRecorded){
		l = new CircleLens3D(modelview, projection, winSize.x, winSize.y, winSize.y * 0.1, lastLensCenter);
	}
	else{
		l = new CircleLens3D(modelview, projection, winSize.x, winSize.y, winSize.y * 0.1, actor->DataCenter());
	}
	lenses->push_back(l);
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
	lenses->push_back(l);
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
		//l = new LineLens3D(lastLensCenter, 0.5); //for VR
		//l = new LineLens3D(lastLensCenter, 0.193742);
	}
	else{
		l = new LineLens3D(actor->DataCenter(), 0.3);
		//l = new LineLens3D(actor->DataCenter(), 0.5); //for VR
		//l = new LineLens3D(actor->DataCenter(), 0.193742);
	}
	lenses->push_back(l);
	//l->justChanged = true; //constructing first, then set justChanged
	actor->UpdateGL();
	actor->SetInteractMode(INTERACT_MODE::ADDING_LENS);
}

void LensRenderable::AddCurveLens()
{
	int2 winSize = actor->GetWindowSize();
	Lens* l = new CurveLens(winSize.y * 0.1, actor->DataCenter());
	lenses->push_back(l);
	//l->justChanged = true; //constructing first, then set justChanged
	actor->UpdateGL();
	actor->SetInteractMode(INTERACT_MODE::ADDING_LENS);
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
		Lens* l = lenses->back();
		if (l->type == LENS_TYPE::TYPE_CURVE) {
			((CurveLens *)l)->AddCtrlPoint(x, y);
		}
		else if (l->type == LENS_TYPE::TYPE_LINE) {
			((LineLens *)l)->ctrlPointScreen1 = make_float2(x, y);
			((LineLens *)l)->ctrlPointScreen2 = make_float2(x, y);
		}	
		break;
	}
	case INTERACT_MODE::TRANSFORMATION:
	{
		if(lenses->size()<1)
			return;
		Lens* l = lenses->back();
		if (l->PointOnLensCenter(x, y, modelview, projection, winSize.x, winSize.y)) {
			actor->SetInteractMode(INTERACT_MODE::MOVE_LENS);
			break;
		}
		else if (l->PointOnOuterBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
			actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE);
			break;
		}
		else if (l->PointOnInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
			actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE);
			std::cout << "hehaherha" << std::endl;
			break;
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
	float3 posMin, posMax;
	actor->GetPosRange(posMin, posMax);

	if (INTERACT_MODE::ADDING_LENS == actor->GetInteractMode()) {
		Lens* l = lenses->back();
		if (l->type == LENS_TYPE::TYPE_LINE) {
			if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE)
			{
				((LineLens*)l)->FinishConstructing(modelview, projection, winSize.x, winSize.y);
			}
			else
			{
				((LineLens3D*)l)->FinishConstructing(modelview, projection, winSize.x, winSize.y, posMin, posMax);
			}
		}
		else if (l->type == LENS_TYPE::TYPE_CURVE) {
			((CurveLens *)l)->FinishConstructing(modelview, projection, winSize.x, winSize.y);
		}
		l->justChanged = true;
		actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
	}
	else {
		if (actor->GetInteractMode() == INTERACT_MODE::MOVE_LENS || actor->GetInteractMode() == INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE || actor->GetInteractMode() == INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE){
			if (lenses->size() > 0){
				lenses->back()->justChanged = true;
			}
		}
		else if (actor->GetInteractMode() == INTERACT_MODE::TRANSFORMATION && changeLensWhenRotateData){
			//this decides whether to relocate the mesh when rotating the data
			if (lenses->size() > 0){
				Lens* l = lenses->back();
				l->justChanged = true;

				if (l->type == LENS_TYPE::TYPE_LINE && ((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE){
					((LineLens3D*)l)->UpdateObjectLineLens(winSize.x, winSize.y, modelview, projection, posMin, posMax);
				}
			}
		}

		if (actor->GetInteractMode() == INTERACT_MODE::MOVE_LENS && isSnapToFeature){
			/*// !!! DON'T DELETE !!!
			//these code will be process later
			GlyphRenderable* glyphRenderable = (GlyphRenderable*)actor->GetRenderable("glyph");
			Lens* l = (*lenses)[lenses->size() - 1];
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
	float3 posMin, posMax;
	actor->GetPosRange(posMin, posMax);
	switch (actor->GetInteractMode())
	{
	case INTERACT_MODE::ADDING_LENS:
	{
		Lens* l = lenses->back();
		if (l->type == LENS_TYPE::TYPE_CURVE){
			((CurveLens *)l)->AddCtrlPoint(x, y);
		}
		else if (l->type == LENS_TYPE::TYPE_LINE){
			((LineLens *)l)->ctrlPointScreen2 = make_float2(x, y);
			if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE){
				((LineLens*)l)->UpdateInfoFromCtrlPoints(modelview, projection, winSize.x, winSize.y);
			}
		}
		break;
	}
	case INTERACT_MODE::MOVE_LENS:
	{
		float3 moveVec = lenses->back()->MoveLens(x, y, modelview, projection, winSize.x, winSize.y);
		if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE && lenses->back()->type == LENS_TYPE::TYPE_LINE){
			((LineLens3D*)lenses->back())->UpdateObjectLineLens(winSize.x, winSize.y, modelview, projection, posMin, posMax);
		}

		if (isSnapToGlyph){
			/*// !!! DON'T DELETE !!!
			//these code will be process later
			DeformGlyphRenderable* glyphRenderable = (DeformGlyphRenderable*)actor->GetRenderable("glyph");
			glyphRenderable->findClosetGlyph(make_float3((*lenses)[pickedLens]->GetCenter()));
			*/
		}
		else if (isSnapToFeature){
			/*// !!! DON'T DELETE !!!
			//these code will be process later			
			GlyphRenderable* glyphRenderable = (GlyphRenderable*)actor->GetRenderable("glyph");
			snapPos;
			int resid=-1;
			glyphRenderable->findClosetFeature(make_float3((*lenses)[pickedLens]->GetCenter()), snapPos, resid);

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
		break;
	}
	case INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE:
	{
		if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE)
			lenses->back()->ChangeLensSize(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);
		else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE){
			lenses->back()->ChangeObjectLensSize(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);

			if (lenses->back()->type == TYPE_LINE){
				((LineLens3D*)(lenses->back()))->UpdateObjectLineLens(winSize.x, winSize.y, modelview, projection, posMin, posMax);
			}
		}
		break;
	}
	case INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE:
	{												   
		if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::SCREEN_SPACE)
			lenses->back()->ChangeFocusRatio(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);
		else if (((DeformGLWidget*)actor)->GetDeformModel() == DEFORM_MODEL::OBJECT_SPACE)
			lenses->back()->ChangeObjectFocusRatio(x, y, lastPt.x, lastPt.y, modelview, projection, winSize.x, winSize.y);
		break;
	}	
	}
	lastPt = make_int2(x, y);
}

bool LensRenderable::MouseWheel(int x, int y, int modifier, int delta)
{
	if (lenses->size() < 1)
		return false;

	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	bool insideAnyLens = false;
	Lens* l = lenses->back();
	if (l->PointInsideInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
		insideAnyLens = true;
		float3 posMin, posMax;
		actor->GetPosRange(posMin, posMax);
		float3 dif = posMax - posMin;
		float coeff = min(min(dif.x, dif.y), dif.z)/10.0/20.0/20.0;
		l->ChangeClipDepth(delta*coeff, &matrix_mv.v[0].x, &matrix_pj.v[0].x);
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

inline bool outOfDomain(float3 p, float3 posMin, float3 posMax)
{
	float3 difmin = p - posMin, difmax = p - posMax;

	return min(difmin.x, min(difmin.y, difmin.z))<0 || max(difmax.x, max(difmax.y, difmax.z))>0;
}

float3 LensRenderable::GetTransferredLeapPos(float3 p)
{
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
	if (lenses->size() > 0){
		ChangeLensCenterbyLeap(lenses->back(), (l + r) * 0.5);
		if (LENS_TYPE::TYPE_CIRCLE == lenses->back()->type){
			((CircleLens*)lenses->back())->radius = length(l-r) * 0.5;
			((CircleLens*)lenses->back())->justChanged = true;
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
		l->MoveLens(pScreen.x, pScreen.y, modelview, projection, winSize.x, winSize.y);
	}

}


void LensRenderable::SlotOneHandChanged(float3 p)
{
	std::cout << "one hand..." << p.x << "," << p.y << "," << p.z<< std::endl;
	if (lenses->size() > 0){
		ChangeLensCenterbyLeap(lenses->back(), p);
		//interaction box
		//https://developer.leapmotion.com/documentation/csharp/devguide/Leap_Coordinate_Mapping.html
		(lenses->back())->justChanged = true;
		actor->UpdateGL();
	}
}


bool LensRenderable::SlotOneHandChangedNew_lc(float3 thumpLeap, float3 indexLeap, float3 middleLeap, float3 ringLeap, float4 &markerPos, float &valRight, float &f)
{
	//currently only work for line lens 3D
	//std::cout << "thumpLeap " << thumpLeap.x << " " << thumpLeap.y << " " << thumpLeap.z << std::endl;
	//std::cout << "indexLeap " << indexLeap.x << " " << indexLeap.y << " " << indexLeap.z << std::endl;


#ifdef USE_OSVR
	float enterPichThr = 35, leavePinchThr = enterPichThr + 10;
#else
	float enterPichThr = 25, leavePinchThr = enterPichThr + 10; //different threshold to avoid shaking
#endif

	float d = length(thumpLeap - indexLeap);

	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);

	float3 curPos = GetTransferredLeapPos(indexLeap);
	//float3 curPos = (GetTransferredLeapPos(thumpLeap) + GetTransferredLeapPos(indexLeap)) / 2;


	markerPos = make_float4(curPos, 1.0);


	if (lenses->size() == 0){
		float3 posMin, posMax;
		actor->GetPosRange(posMin, posMax);
		if (actor->GetInteractMode() == INTERACT_MODE::TRANSFORMATION && d < enterPichThr && !outOfDomain(curPos, posMin, posMax)){

			Lens* l = new LineLens3D(actor->DataCenter(), 0.3);
			l->isConstructedFromLeap = true;

			lenses->push_back(l);
			//l->justChanged = true; //constructing first, then set justChanged
			actor->UpdateGL();
			actor->SetInteractMode(INTERACT_MODE::ADDING_LENS);
			((LineLens3D *)l)->ctrlPoint3D1 = curPos;
			((LineLens3D *)l)->ctrlPoint3D2 = curPos;
			valRight = 1;
			return true;
		}
		else{
			//add global rotation?

			if (outOfDomain(curPos, posMin, posMax)){
				return false;
			}
			else{
				return true;
			}
			return true;
		}
	}
	else{	//lenses->size()>0
		LineLens3D* l = (LineLens3D*)lenses->back();


#ifdef USE_OSVR
		float d2 = length(middleLeap - indexLeap) / 1.1;
		float d3 = length(middleLeap - ringLeap) / 1.1;
#else
		float d2 = length(middleLeap - indexLeap) / 1.2;
		float d3 = length(middleLeap - ringLeap) / 1.2;
#endif



		switch (actor->GetInteractMode())
		{
		case INTERACT_MODE::ADDING_LENS:
		{
			if (d < leavePinchThr){
				l->ctrlPoint3D2 = curPos;
				return true;
			}
			else {
				float3 posMin, posMax;
				actor->GetPosRange(posMin, posMax);
				l->FinishConstructing3D(modelview, projection, winSize.x, winSize.y, posMin, posMax);

				l->justChanged = true;

				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				return true;
			}
			break;
		}
		case INTERACT_MODE::TRANSFORMATION:
		{
			if (d < enterPichThr){
				valRight = 1;
				if (l->PointOnLensCenter3D(curPos, modelview, projection, winSize.x, winSize.y)){
					actor->SetInteractMode(INTERACT_MODE::MOVE_LENS);
					prevPos = curPos;
					prevPointOfLens = l->c;
				}
				else if (l->PointOnOuterBoundaryWallMajorSide3D(curPos, modelview, projection, winSize.x, winSize.y)){
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
			else if (d2<enterPichThr && d3<enterPichThr){
				actor->SetInteractMode(INTERACT_MODE::CHANGING_FORCE);
				preForce = f;
				prevPos = curPos;

				valRight = 2;
			}
			else{
				valRight = 0;
				float3 posMin, posMax;
				actor->GetPosRange(posMin, posMax);
				if (outOfDomain(curPos, posMin, posMax)){
					return false;
				}
				else{
					if (l->PointOnLensCenter3D(curPos, modelview, projection, winSize.x, winSize.y)){
						highlightingCenter = true;
						highlightingMajorSide = false;
						highlightingMinorSide = false;
						highlightingCuboidFrame = false;
					}
					else if (l->PointOnOuterBoundaryWallMajorSide3D(curPos, modelview, projection, winSize.x, winSize.y)){
						highlightingCenter = false;
						highlightingMajorSide = true;
						highlightingMinorSide = false;
						highlightingCuboidFrame = false;
					}
					else if (l->PointOnOuterBoundaryWallMinorSide3D(curPos, modelview, projection, winSize.x, winSize.y)){
						highlightingCenter = false;
						highlightingMajorSide = false;
						highlightingMinorSide = true;
						highlightingCuboidFrame = false;
					}
					else if (l->PointInCuboidRegion3D(curPos, modelview, projection, winSize.x, winSize.y)){
						highlightingCenter = false;
						highlightingMajorSide = false;
						highlightingMinorSide = false;
						highlightingCuboidFrame = true;
					}
					else{
						highlightingCenter = false;
						highlightingMajorSide = false;
						highlightingMinorSide = false;
						highlightingCuboidFrame = false;
					}
				}
				return true;
			}
			break;
		}
		case INTERACT_MODE::CHANGING_FORCE:
		{
			if (d2 > leavePinchThr || d3 > leavePinchThr){
				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				valRight = 0;
			}
			else{
				float3 c = l->c;
				f = preForce + (abs(dot(curPos - c, l->minorAxisGlobal)) - abs(dot(prevPos - c, l->minorAxisGlobal))) * 2*3;
				if (f < 0)
					f = 0;
				//send back new force
			}
			return true;
			break;
		}
		case INTERACT_MODE::MOVE_LENS:
		{
			if (d > leavePinchThr){
				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				(lenses->back())->justChanged = true;
			}
			else{
				float3 moveDir = curPos - prevPos;
				ChangeLensCenterbyTransferredLeap(lenses->back(), prevPointOfLens + moveDir);
			}
			return true;
			break;
		}
		case INTERACT_MODE::MODIFY_LENS_FOCUS_SIZE:
		{
			if (d > leavePinchThr){
				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				(lenses->back())->justChanged = true;
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
			break;
		}
		case INTERACT_MODE::MODIFY_LENS_TRANSITION_SIZE:
		{
			float3 moveDir = curPos - prevPos;
			float3 pp = prevPointOfLens + moveDir;
			float newMinorDis = abs(dot(pp - l->c, l->minorAxisGlobal));

			if (d > leavePinchThr && newMinorDis > l->lSemiMinorAxisGlobal){
				actor->SetInteractMode(INTERACT_MODE::TRANSFORMATION);
				(lenses->back())->justChanged = true;
				//highlightingMinorSide = false;
			}
			else{
				l->focusRatio = l->lSemiMinorAxisGlobal / newMinorDis;
			}
			return true;
			break;
		}
		default:
		{
				   //do nothing
				   return false;
		}
		}
	}
}

//////////////////////////////////	//for keyboard /////////////////////

void LensRenderable::adjustOffset(){
	if (lenses->size() < 1)
		return;
	Lens* l = lenses->back();

	if (l->type == LENS_TYPE::TYPE_CURVE) {
		((CurveLens*)l)->adjustOffset();

	}
};

void LensRenderable::RefineLensBoundary(){
	if (lenses->size() < 1)
		return;
	Lens* l = lenses->back();
	if (l->type == LENS_TYPE::TYPE_CURVE) {
		((CurveLens*)l)->RefineLensBoundary();

	}
};



void LensRenderable::SlotDelLens()
{
	activedCursors = 0;
	if (lenses->size() > 0){
		lastLensCenter = make_float3(lenses->back()->GetCenter());
		lastLensCenterRecorded = true;
		lenses->pop_back();
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
	//for (int i = 0; i < lenses->size(); i++) {
	//	Lens* l = (*lenses)[i];
	//	if (l->PointInsideInnerBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
	//		pickedLens = i;
	//		break;
	//	}
	//}
	//if (pickedLens > -1){
	if (lenses->size()<1)
		return;
	if (INTERACT_MODE::MODIFY_LENS_DEPTH == actor->GetInteractMode()){
		//actor->SetInteractMode(INTERACT_MODE::MODIFY_LENS_DEPTH);
		//std::cout << "totalScaleFactor:" << totalScaleFactor << std::endl;
		float scaleFactor = totalScaleFactor > 1 ? 1 : -1;
		lenses->back()->ChangeClipDepth(scaleFactor, &matrix_mv.v[0].x, &matrix_pj.v[0].x);
		lenses->back()->justChanged = true;
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
	if (lenses->size() == 0) return;
	lenses->back()->ChangeClipDepth(v, &matrix_mv.v[0].x, &matrix_pj.v[0].x);

	//lenses->back()->justChanged = true;
	actor->UpdateGL();
}

bool LensRenderable::InsideALens(int x, int y)
{
	if (lenses->size() < 1)
		return false;
	
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	bool ret = false;
	Lens* l = lenses->back();
		if (l->PointInsideOuterBoundary(x, y, modelview, projection, winSize.x, winSize.y)) {
			ret = true;
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
	if (lenses->size() > 0) {
		lenses->back()->ChangeLensTwoFingers(p1, p2, modelview, projection, winSize.x, winSize.y);
		lenses->back()->justChanged = true;
		actor->UpdateGL();
	}
}

bool LensRenderable::TwoPointsInsideALens(int2 p1, int2 p2)
{
	if (lenses->size() < 1)
		return false;
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	bool ret = false;
	Lens* l = lenses->back();
		if (l->PointInsideInnerBoundary(p1.x, p1.y, modelview, projection, winSize.x, winSize.y)
			&& l->PointInsideInnerBoundary(p2.x, p2.y, modelview, projection, winSize.x, winSize.y)) {
			ret = true;
	}
	return ret;
}

bool LensRenderable::OnLensInnerBoundary(int2 p1, int2 p2)
{
	if (lenses->size() < 1)
		return false;
	int2 winSize = actor->GetWindowSize();
	GLfloat modelview[16];
	GLfloat projection[16];
	actor->GetModelview(modelview);
	actor->GetProjection(projection);
	bool ret = false;
	Lens* l = lenses->back();
		if (l->PointOnInnerBoundary(p1.x, p1.y, modelview, projection, winSize.x, winSize.y)
			&& l->PointOnInnerBoundary(p2.x, p2.y, modelview, projection, winSize.x, winSize.y)) {
			ret = true;
	}
	return ret;
}


void LensRenderable::SaveState(const char* filename)
{
	std::ofstream myfile;
	myfile.open(filename);

	if (lenses->size() == 0)
		return;
	LineLens3D* l = (LineLens3D*)lenses->back();
	myfile << l->c.x << " " << l->c.y << " " << l->c.z << std::endl;
	
	myfile << l->ctrlPointScreen1.x << " " << l->ctrlPointScreen1.y << std::endl;
	myfile << l->ctrlPointScreen2.x << " " << l->ctrlPointScreen2.y << std::endl;
	myfile << l->focusRatio;
	myfile.close();
}

void LensRenderable::LoadState(const char* filename)
{
	std::ifstream ifs(filename, std::ifstream::in);
	if (ifs.is_open()) {
		if (lenses->size() == 0)
			return;
		LineLens3D* l = (LineLens3D*)lenses->back();

		float3 _c;
		ifs >> _c.x >> _c.y >> _c.z;
		l->SetCenter(_c);

		ifs >> l->ctrlPointScreen1.x;
		ifs >> l->ctrlPointScreen1.y;
		ifs >> l->ctrlPointScreen2.x;
		ifs >> l->ctrlPointScreen2.y;

		ifs >> l->focusRatio;

		ifs.close();
		l->justChanged = true;
	}
}
