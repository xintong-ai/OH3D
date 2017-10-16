#include "BoxRenderable.h"
#include <QOpenGLWidget>

BoxRenderable::BoxRenderable(int3 d)
{
	//	SetVecReader(r);
	dim.x = d.x;
	dim.y = d.y;
	dim.z = d.z;
	pos = make_float3(0, 0, 0);
}

BoxRenderable::BoxRenderable(float x, float y, float z, float nx, float ny, float nz)
{
	pos = make_float3(x, y, z);
	dim = make_float3(nx, ny, nz);
}

BoxRenderable::BoxRenderable(float3 _pos, float3 _dim)
{
	pos = make_float3(_pos.x, _pos.y, _pos.z);
	dim = make_float3(_dim.x, _dim.y, _dim.z);
}

void BoxRenderable::draw(float modelview[16], float projection[16])
{
	if (!visible)
		return;
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glTranslatef(pos.x, pos.y, pos.z);
	float sx = -0.5, sy = -0.5, sz = -0.5;
	float bx = dim.x - 1 + 0.5, by = dim.y - 1 + 0.5, bz = dim.z - 1 + 0.5;

	glBegin(GL_LINES);
	glVertex3f(sx, sy, sz);
	glVertex3f(bx, sy, sz);

	glVertex3f(bx, sy, sz);
	glVertex3f(bx, by, sz);

	glVertex3f(bx, by, sz);
	glVertex3f(sx, by, sz);

	glVertex3f(sx, by, sz);
	glVertex3f(sx, sy, sz);

	//////
	glVertex3f(sx, sy, sz);
	glVertex3f(sx, sy, bz);

	glVertex3f(bx, sy, sz);
	glVertex3f(bx, sy, bz);

	glVertex3f(bx, by, sz);
	glVertex3f(bx, by, bz);

	glVertex3f(sx, by, sz);
	glVertex3f(sx, by, bz);

	//////
	glVertex3f(sx, sy, bz);
	glVertex3f(bx, sy, bz);

	glVertex3f(bx, sy, bz);
	glVertex3f(bx, by, bz);

	glVertex3f(bx, by, bz);
	glVertex3f(sx, by, bz);

	glVertex3f(sx, by, bz);
	glVertex3f(sx, sy, bz);

	glEnd();
	glPopMatrix();
}
