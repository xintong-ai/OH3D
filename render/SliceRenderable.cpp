#include "SliceRenderable.h"
#include "GLMatrixManager.h"
#include "GLWidget.h"


#include <iostream>
#include <algorithm>
using namespace std;


void SliceRenderable::init()
{
}


void SliceRenderable::draw(float modelview[16], float projection[16])
{
	RecordMatrix(modelview, projection);

	if (!visible)
		return;
	int3 size = volume->size;

	//glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);

	//	glDisable(GL_DEPTH_TEST);
	// glDepthFunc(GL_ALWAYS);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glColor4f(0.89f, 0.89f, 0.26f, 1);

	glPointSize(16);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(projection);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(modelview);

	glBegin(GL_POINTS);

	int j = 45;
	for (int k = 0; k < size.z; k++){
		for (int i = 0; i < size.x; i++){
			int ind = k*size.x*size.y + j*size.x + i;
			if (volume->values[ind]>0.5){
				glVertex3f(i, j, k);
			}
		}
	}

	glEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glDisable(GL_BLEND);
	glPopAttrib();
	//glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

}
