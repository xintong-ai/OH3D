#include "GridRenderable.h"
#include "GlyphRenderable.h"
#include "glwidget.h"

GridRenderable::GridRenderable(int n)
{
	gridResolution.x = n;

	//winWidth
}

void GridRenderable::UpdateGrid()
{
	((GlyphRenderable*)actor->GetRenderable("glyph"))->DisplacePoints(grid);
}

void GridRenderable::resize(int width, int height)
{
	float side = width / (gridResolution.x - 1);
	gridResolution.y = ceil(height / side) + 1;
	float x, y;
	grid.clear();
	for (int j = 0; j < gridResolution.y; j++) {
		for (int i = 0; i < gridResolution.x; i++) {
			if (i < gridResolution.x - 1)
				x = i * side;
			else
				x = width - 1;
			if (j < gridResolution.y - 1)
				y = j * side;
			else
				y = height - 1;
			grid.push_back(make_float2(x, y));
		}
	}
	UpdateGrid();
	mesh.clear();
	for (int j = 0; j < gridResolution.y - 1; j++) {
		for (int i = 0; i < gridResolution.x - 1; i++) {
			mesh.push_back(grid[j * gridResolution.x + i]);
			mesh.push_back(grid[j * gridResolution.x + i + 1]);
			mesh.push_back(grid[(j + 1) * gridResolution.x + i + 1]);
			mesh.push_back(grid[(j + 1) * gridResolution.x + i]);
		}
	}
}

void GridRenderable::draw(float modelview[16], float projection[16])
{
	int2 winSize = actor->GetWindowSize();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, winSize.x - 1, 0.0, winSize.y - 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glPushAttrib(GL_LINE_BIT | GL_CURRENT_BIT);
	//glLineWidth(1);
	glColor3f(0.8f, 0.8f, 0.8f);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(2, GL_FLOAT, 0, &mesh[0].x);
	glDrawArrays(GL_QUADS, 0, mesh.size());
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glPopAttrib();
	//restore the original 3D coordinate system
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}
