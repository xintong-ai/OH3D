#ifndef GLYPH_RENDERABLE_H
#define GLYPH_RENDERABLE_H

#include "Renderable.h"
#include <memory>
class Displace;
class ShaderProgram;
class QOpenGLContext;

class GlyphRenderable: public Renderable
{
	Q_OBJECT

protected:
	std::vector<float4> pos;
	std::shared_ptr<Displace> displace;
	std::vector<float> glyphSizeScale;
	std::vector<float> glyphBright;
	float glyphSizeAdjust = 0.5;
	int snappedGlyphIdx = -1;
	ShaderProgram* glProg = nullptr;
	//bool displaceOn = true;
	void ComputeDisplace();
	void mouseMove(int x, int y, int modifier) override;
	void resize(int width, int height) override;
	GlyphRenderable(std::vector<float4>& _pos);

	
public:
	~GlyphRenderable();
	void RecomputeTarget();
	void DisplacePoints(std::vector<float2>& pts);
	virtual void LoadShaders(ShaderProgram*& shaderProg) = 0;
	virtual void DrawWithoutProgram(float modelview[16], float projection[16], ShaderProgram* sp) = 0;
	//void SetDispalceOn(bool b) { displaceOn = b; }
	float3 findClosetGlyph(float3 aim);

	void mousePress(int x, int y, int modifier) override;


	//virtual void drawPicking(float modelview[16], float projection[16]) = 0;
	unsigned int vbo_vert_picking;
	int snappedGlyphId = -1;
	ShaderProgram *glPickingProg;
	int numVerticeOfGlyph = 0;
	void initForPicking(int nv, float* vertex);
	void drawPicking(float modelview[16], float projection[16]);

	bool isPicking = false;
	unsigned int framebuffer, renderbuffer[2];
	int xMouse, yMouse;

	float mymv[16], myp[16];
	void myRecordMatrix(float modelview[16], float projection[16])
	{
		memcpy(mymv, modelview, sizeof(float)* 16);
		memcpy(myp, projection, sizeof(float)* 16);

	};

public slots:
	void SlotGlyphSizeAdjustChanged(int v);
};
#endif