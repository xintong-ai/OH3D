#include "helper_math.h"
#include "TransformFunc.h"

#include "TraceRenderable.h"

#include "glwidget.h"
#include "Particle.h"


//removing the following lines will cause runtime error
#ifdef WIN32
#include "windows.h"
#endif
#define qgl	QOpenGLContext::currentContext()->functions()
#include <QOpenGLFunctions>
#include <QOpenGLVertexArrayObject>
#include "ShaderProgram.h"

using namespace std;

vector<float2> convex_hull(vector<float2> P);


TraceRenderable::TraceRenderable(std::vector<std::shared_ptr<Particle>> _particle)
{
	particleSet = _particle;
}


TraceRenderable::~TraceRenderable()
{
}


void TraceRenderable::init()
{

}




void TraceRenderable::draw(float modelview[16], float projection[16])
{
	if (!visible)
		return;
	RecordMatrix(modelview, projection);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(projection);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glLoadMatrixf(modelview);

	glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);


	glLineWidth(3);
	for (int i = 0; i < particleSet.size(); i++){
		glColor3f(0.0, 1.0f, 0.0);
		glBegin(GL_LINE_STRIP);
		for (int j = 0; j < particleSet[i]->numParticles; j++){
			glVertex3f(particleSet[i]->pos[j].x, particleSet[i]->pos[j].y, particleSet[i]->pos[j].z);
		}
		glEnd();
	}
	
		
	glPopAttrib();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glPopAttrib();

	
}

float cross(const float2 &O, const float2 &A, const float2 &B)
{
	return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

