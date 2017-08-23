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


void DeformFrameRenderable::draw(float modelview[16], float projection[16])
{
	RecordMatrix(modelview, projection);

	if (!visible)
		return;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(projection);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(modelview);


	glColor4f(0.89f, 0.29f, 0.26f, 0.8f);
	glLineWidth(4);
	//if (processor->hasOpenAnimeStarted || processor->hasCloseAnimeStarted)
	{

		float3 zaxisn = normalize(processor->tunnelEnd - processor->tunnelStart);
		float3 yaxis = processor->rectVerticalDir;
		float3 xaxis = normalize(cross(zaxisn, yaxis));
		float deformationScale = processor->deformationScale;
		float deformationScaleVertical = processor->deformationScaleVertical;

		float3 p0 = processor->tunnelStart - xaxis*deformationScale - yaxis*deformationScaleVertical;
		float3 p1 = processor->tunnelStart + xaxis*deformationScale - yaxis*deformationScaleVertical;
		float3 p2 = processor->tunnelStart + xaxis*deformationScale + yaxis*deformationScaleVertical;
		float3 p3 = processor->tunnelStart - xaxis*deformationScale + yaxis*deformationScaleVertical;

		//cout <<endl;
		glBegin(GL_LINE_LOOP);
		glVertex3fv(&(p0.x));
		glVertex3fv(&(p1.x));
		glVertex3fv(&(p2.x));
		glVertex3fv(&(p3.x));
		glEnd();

		float3 p4 = processor->tunnelEnd - xaxis*deformationScale - yaxis*deformationScaleVertical;
		float3 p5 = processor->tunnelEnd + xaxis*deformationScale - yaxis*deformationScaleVertical;
		float3 p6 = processor->tunnelEnd + xaxis*deformationScale + yaxis*deformationScaleVertical;
		float3 p7 = processor->tunnelEnd - xaxis*deformationScale + yaxis*deformationScaleVertical;

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

	bool blockFurtherObjects = false;
	if (blockFurtherObjects){
		float3 zaxisn = normalize(processor->tunnelEnd - processor->tunnelStart);
		float3 yaxis = processor->rectVerticalDir;
		float3 xaxis = normalize(cross(zaxisn, yaxis));
		float deformationScale = processor->deformationScale;
		float deformationScaleVertical = processor->deformationScaleVertical;

		glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
		float3 pp4 = processor->tunnelEnd - xaxis*deformationScale * 100 - yaxis*deformationScaleVertical * 100;
		float3 pp5 = processor->tunnelEnd + xaxis*deformationScale * 100 - yaxis*deformationScaleVertical * 100;
		float3 pp6 = processor->tunnelEnd + xaxis*deformationScale * 100 + yaxis*deformationScaleVertical * 100;
		float3 pp7 = processor->tunnelEnd - xaxis*deformationScale * 100 + yaxis*deformationScaleVertical * 100;

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



