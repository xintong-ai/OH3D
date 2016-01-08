#ifndef GLYPH_RENDERABLE_H
#define GLYPH_RENDERABLE_H

#include "Renderable.h"
#include <QObject>
#include <Displace.h>

class ShaderProgram;
class QOpenGLVertexArrayObject;
class GLSphere;
class SphereRenderable :public Renderable, public QObject
{
	Q_OBJECT
public:
	void init() override;
	void draw(float modelview[16], float projection[16]) override;
	void UpdateData() override;
	SphereRenderable(float3* _spherePos, int _sphereCnt, float* _sphereSize = nullptr);// { spherePos = _spherePos; sphereCnt = _sphereCnt; sphereSize = _sphereSize; }

	//void SetVolumeDim(int x, int y, int z){ dataDim[0] = x; dataDim[1] = y; dataDim[2] = z; }
	void SetVolRange(float3 _dataMin, float3 _dataMax) { dataMin = _dataMin; dataMax = _dataMax; }
private:
	float3* spherePos = nullptr;
	float* sphereSize = nullptr;
	int sphereCnt = 0;
	void GenVertexBuffer(int nv, float* vertex);
	void LoadShaders();
	unsigned int vbo_vert;
	GLSphere* glyphMesh;
	ShaderProgram *glProg;

	QOpenGLVertexArrayObject* m_vao;
	bool updated = false;
	//int dataMin[3];
	float3 dataMin, dataMax;

	float3 DataCenter();// { return (dataMin + dataMax) * 0.5; }
	Displace displace;
};
#endif //GLYPH_RENDERABLE_H
