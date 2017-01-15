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

	if (INTERACT_MODE::MOVE_LENS == actor->GetInteractMode()){
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




//////////////////////////////////	//for qt button /////////////////////

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
		l = new CircleLens3D(modelview, projection, winSize.x, winSize.y, winSize.y * 0.15, lastLensCenter);
	}
	else{
		l = new CircleLens3D(modelview, projection, winSize.x, winSize.y, winSize.y * 0.15, actor->DataCenter());
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
		l = new LineLens(lastLensCenter, lastLensRatio);
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
		//l = new LineLens3D(lastLensCenter, 0.222048);// 0.3);
		//l = new LineLens3D(lastLensCenter, 0.5); //for VR
		l = new LineLens3D(lastLensCenter, lastLensRatio);
	}
	else{
		//l = new LineLens3D(actor->DataCenter(), 0.222048);// 0.3);
		//l = new LineLens3D(actor->DataCenter(), 0.5); //for VR
		l = new LineLens3D(actor->DataCenter(), 0.193742);
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



void LensRenderable::DelLens()
{
	if (lenses->size() > 0){
		lastLensCenter = make_float3(lenses->back()->GetCenter());
		if (lenses->back()->type == LENS_TYPE::TYPE_LINE){
			lastLensRatio = ((lenses->back()))->focusRatio;
		}

		lastLensCenterRecorded = true;
		lenses->pop_back();
	}
	actor->UpdateGL();
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
		//l->axisRatio = l->focusRatio / 3.0;

		ifs.close();
		l->justChanged = true;
	}
}
