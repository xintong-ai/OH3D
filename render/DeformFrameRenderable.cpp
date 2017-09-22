#include "DeformFrameRenderable.h"
#include "GLMatrixManager.h"
#include "GLWidget.h"
#include "PositionBasedDeformProcessor.h"


#include <iostream>
#include <algorithm>
using namespace std;

void DeformFrameRenderable::init()
{
}

void DeformFrameRenderable::drawCuboidModel(float modelview[16], float projection[16])
{
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(projection);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(modelview);

	bool drawFrame = true;
	if (drawFrame){
		glColor4f(0.89f, 0.29f, 0.26f, 0.8f);
		glLineWidth(4);
		{

			float3 zaxisn = normalize(processor->getTunnelEnd() - processor->getTunnelStart());
			float3 yaxis = processor->getRectVerticalDir();
			float3 xaxis = normalize(cross(zaxisn, yaxis));
			float deformationScale = processor->getDeformationScale();
			float deformationScaleVertical = processor->getDeformationScaleVertical();

			float3 p0 = processor->getTunnelStart() - xaxis*deformationScale - yaxis*deformationScaleVertical;
			float3 p1 = processor->getTunnelStart() + xaxis*deformationScale - yaxis*deformationScaleVertical;
			float3 p2 = processor->getTunnelStart() + xaxis*deformationScale + yaxis*deformationScaleVertical;
			float3 p3 = processor->getTunnelStart() - xaxis*deformationScale + yaxis*deformationScaleVertical;

			//cout <<endl;
			glBegin(GL_LINE_LOOP);
			glVertex3fv(&(p0.x));
			glVertex3fv(&(p1.x));
			glVertex3fv(&(p2.x));
			glVertex3fv(&(p3.x));
			glEnd();

			float3 p4 = processor->getTunnelEnd() - xaxis*deformationScale - yaxis*deformationScaleVertical;
			float3 p5 = processor->getTunnelEnd() + xaxis*deformationScale - yaxis*deformationScaleVertical;
			float3 p6 = processor->getTunnelEnd() + xaxis*deformationScale + yaxis*deformationScaleVertical;
			float3 p7 = processor->getTunnelEnd() - xaxis*deformationScale + yaxis*deformationScaleVertical;

			glBegin(GL_LINE_LOOP);
			glVertex3fv(&(p4.x));
			glVertex3fv(&(p5.x));
			glVertex3fv(&(p6.x));
			glVertex3fv(&(p7.x));
			glEnd();

			glBegin(GL_LINES);
			glVertex3fv(&(p0.x));
			glVertex3fv(&(p4.x));
			glVertex3fv(&(p1.x));
			glVertex3fv(&(p5.x));
			glVertex3fv(&(p2.x));
			glVertex3fv(&(p6.x));
			glVertex3fv(&(p3.x));
			glVertex3fv(&(p7.x));
			glEnd();
		}
		glLineWidth(1); //restore
	}
	
	bool drawCentralPlane = true;
	if (drawCentralPlane){
		glColor4f(0.0f, 112.0/256, 192.0/256, 0.5f);
		float3 zaxisn = normalize(processor->getTunnelEnd() - processor->getTunnelStart());
		float3 yaxis = processor->getRectVerticalDir();
		float3 xaxis = normalize(cross(zaxisn, yaxis));
		float deformationScale = processor->getDeformationScale();
		float deformationScaleVertical = processor->getDeformationScaleVertical();

		float3 p0 = processor->getTunnelStart() - yaxis*deformationScaleVertical;

		float3 p2 = processor->getTunnelStart() + yaxis*deformationScaleVertical;

		float3 p4 = processor->getTunnelEnd() - yaxis*deformationScaleVertical;

		float3 p6 = processor->getTunnelEnd() + yaxis*deformationScaleVertical;


		glBegin(GL_QUADS);
		glVertex3fv(&(p0.x));
		glVertex3fv(&(p2.x));
		glVertex3fv(&(p6.x));
		glVertex3fv(&(p4.x));
		glEnd();
	}

	bool blockFurtherObjects = false;
	if (blockFurtherObjects){
		float3 zaxisn = normalize(processor->getTunnelEnd() - processor->getTunnelStart());
		float3 yaxis = processor->getRectVerticalDir();
		float3 xaxis = normalize(cross(zaxisn, yaxis));
		float deformationScale = processor->getDeformationScale();
		float deformationScaleVertical = processor->getDeformationScaleVertical();

		//glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
		//glColor4f(0.7f, 0.7f, 0.7f, 1.0f);
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

		float3 pp4 = processor->getTunnelEnd() + zaxisn * 0.01 - xaxis*deformationScale * 100 - yaxis*deformationScaleVertical * 100;
		float3 pp5 = processor->getTunnelEnd() + zaxisn * 0.01 + xaxis*deformationScale * 100 - yaxis*deformationScaleVertical * 100;
		float3 pp6 = processor->getTunnelEnd() + zaxisn * 0.01 + xaxis*deformationScale * 100 + yaxis*deformationScaleVertical * 100;
		float3 pp7 = processor->getTunnelEnd() + zaxisn * 0.01 - xaxis*deformationScale * 100 + yaxis*deformationScaleVertical * 100;

		glBegin(GL_TRIANGLES);
		glVertex3fv(&(pp4.x));
		glVertex3fv(&(pp5.x));
		glVertex3fv(&(pp6.x));
		glVertex3fv(&(pp4.x));
		glVertex3fv(&(pp7.x));
		glVertex3fv(&(pp6.x));
		glEnd();
	}

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void DeformFrameRenderable::drawCircileModel(float modelview[16], float projection[16])
{
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(projection);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(modelview);

	glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
	glLineWidth(30);
	{
		float3 zaxisn = normalize(processor->getTunnelEnd() - processor->getTunnelStart());
		float3 pstart = processor->getTunnelStart();// -zaxisn * 500;
		float3 p0 = processor->getTunnelStart();
		float3 p1 = processor->getTunnelEnd();

		glBegin(GL_LINES);
		glVertex3fv(&(p0.x));
		glVertex3fv(&(p1.x));
		glEnd();
	}
	glLineWidth(1); //restore

	bool drawLineFrame = true;
	if (drawLineFrame){
		glLineWidth(4);
		glColor4f(0.89f, 0.29f, 0.26f, 0.8f);
		{
			float3 zaxisn = normalize(processor->getTunnelEnd() - processor->getTunnelStart());
			float3 xaxis = normalize(cross(zaxisn, cross(zaxisn, make_float3(1, 0, 0)) + cross(zaxisn, make_float3(0, 1, 0))));
			float3 yaxis = normalize(cross(zaxisn, xaxis));
			float radius = processor->radius; ///!!!!!!!!!!! NOTE !!!!!!halved the value for easier observation
			const int n = 16;
			vector<float3> top(n);
			vector<float3> bottom(n);

			for (int i = 0; i < n; i++){
				float angle = 2 * 3.1415926 / n * i;
				top[i] = processor->getTunnelEnd() + xaxis*radius * cos(angle) + yaxis*radius * sin(angle);
				bottom[i] = processor->getTunnelStart() + xaxis*radius * cos(angle) + yaxis*radius * sin(angle);
			}

			glBegin(GL_LINE_LOOP);
			for (int i = 0; i < n; i++){
				glVertex3fv(&(top[i].x));

			}
			glEnd();
			glBegin(GL_LINE_LOOP);
			for (int i = 0; i < n; i++){
				glVertex3fv(&(bottom[i].x));

			}
			glEnd();

			glBegin(GL_LINES);
			for (int i = 0; i < n; i++){
				glVertex3fv(&(top[i].x));
				glVertex3fv(&(bottom[i].x));
			}
			glEnd();
		}

		/*
		{
		float3 zaxisn = normalize(processor->getTunnelEnd() - processor->getTunnelStart());
		float3 xaxis = normalize(cross(zaxisn, cross(zaxisn, make_float3(1, 0, 0)) + cross(zaxisn, make_float3(0, 1, 0))));
		float3 yaxis = normalize(cross(zaxisn, xaxis));
		float radius = processor->radius*0.5; ///!!!!!!!!!!! NOTE !!!!!!halved the value for easier observation
		const int n = 16;
		vector<float3> top(n);
		vector<float3> bottom(n);

		for (int i = 0; i < n; i++){
		float angle = 2 * 3.1415926 / n * i;
		top[i] = processor->getTunnelEnd() + xaxis*radius * cos(angle) + yaxis*radius * sin(angle);
		bottom[i] = processor->getTunnelStart() + xaxis*radius * cos(angle) + yaxis*radius * sin(angle);
		}

		glBegin(GL_LINE_LOOP);
		for (int i = 0; i < n; i++){
		glVertex3fv(&(top[i].x));

		}
		glEnd();
		glBegin(GL_LINE_LOOP);
		for (int i = 0; i < n; i++){
		glVertex3fv(&(bottom[i].x));

		}
		glEnd();

		glBegin(GL_LINES);
		for (int i = 0; i < n; i++){
		glVertex3fv(&(top[i].x));
		glVertex3fv(&(bottom[i].x));
		}
		glEnd();
		}
		*/
		glLineWidth(1); //restore
	}
	else{
		glColor4f(0.89f, 0.29f, 0.26f, 0.8f);
		{
			float3 zaxisn = normalize(processor->getTunnelEnd() - processor->getTunnelStart());
			float3 xaxis = normalize(cross(zaxisn, cross(zaxisn, make_float3(1, 0, 0)) + cross(zaxisn, make_float3(0, 1, 0))));
			float3 yaxis = normalize(cross(zaxisn, xaxis));
			float radius = processor->radius; ///!!!!!!!!!!! NOTE !!!!!!halved the value for easier observation
			const int n = 16;
			vector<float3> top(n);
			vector<float3> bottom(n);

			for (int i = 0; i < n; i++){
				float angle = 2 * 3.1415926 / n * i;
				top[i] = processor->getTunnelEnd() + xaxis*radius * cos(angle) + yaxis*radius * sin(angle);
				bottom[i] = processor->getTunnelStart() + xaxis*radius * cos(angle) + yaxis*radius * sin(angle);
			}

			glBegin(GL_QUADS);
			for (int i = 0; i < n; i++){
				glVertex3fv(&(top[i].x));
				glVertex3fv(&(bottom[i].x));
				glVertex3fv(&(bottom[(i + 1) % n].x));
				glVertex3fv(&(top[(i + 1) % n].x));
			}
			glEnd();
		}
	}


	bool blockFurtherObjects = false;
	if (blockFurtherObjects){
		float3 zaxisn = normalize(processor->getTunnelEnd() - processor->getTunnelStart());
		float3 yaxis = processor->getRectVerticalDir();
		float3 xaxis = normalize(cross(zaxisn, yaxis));
		float deformationScale = processor->getDeformationScale();
		float deformationScaleVertical = processor->getDeformationScaleVertical();

		glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
		float3 pp4 = processor->getTunnelEnd() - xaxis*deformationScale * 100 - yaxis*deformationScaleVertical * 100;
		float3 pp5 = processor->getTunnelEnd() + xaxis*deformationScale * 100 - yaxis*deformationScaleVertical * 100;
		float3 pp6 = processor->getTunnelEnd() + xaxis*deformationScale * 100 + yaxis*deformationScaleVertical * 100;
		float3 pp7 = processor->getTunnelEnd() - xaxis*deformationScale * 100 + yaxis*deformationScaleVertical * 100;

		glBegin(GL_TRIANGLES);
		glVertex3fv(&(pp4.x));
		glVertex3fv(&(pp5.x));
		glVertex3fv(&(pp6.x));
		glVertex3fv(&(pp4.x));
		glVertex3fv(&(pp7.x));
		glVertex3fv(&(pp6.x));
		glEnd();
	}

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

void DeformFrameRenderable::draw(float modelview[16], float projection[16])
{
	RecordMatrix(modelview, projection);

	if (!visible)
		return;
	
	if (processor->getSystemState() == SYSTEM_STATE::ORIGINAL){
		return;
	}

	if (processor->getShapeModel() == SHAPE_MODEL::CUBOID){
		drawCuboidModel(modelview, projection);
	}
	else if (processor->getShapeModel() == SHAPE_MODEL::CIRCLE){
		drawCircileModel(modelview, projection);
	}
}



